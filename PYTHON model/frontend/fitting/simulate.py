from scipy.optimize import curve_fit
import scipy.stats as stats
import os
import multiprocessing as mp
import argparse
from block_timer.timer import Timer

from backend.functions import *
from backend.models import *
from backend.experiments import *

###############################################################################
#                                  Setup
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument( "-e", '--experiment', type=str,
                     choices=[ "hp_sine", "hp_pulsed", "oblea_sine", "oblea_pulsed", "miao", "jo" ],
                     help="The input shape to use." )
parser.add_argument( '-s', '--solvers', nargs="+", default=[ "LSODA" ],
                     help="The solvers to use to simulate the model's evolution." )
parser.add_argument( '--video', dest="video", action="store_true",
                     help="Generate video of the simulation." )
parser.set_defaults( new=False )
args = parser.parse_args()

experiments = {
        "hp_sine"     : hp_labs_sine,
        "hp_pulsed"   : hp_labs_pulsed,
        "oblea_sine"  : oblea_sine,
        "oblea_pulsed": oblea_pulsed,
        "miao"        : miao,
        "jo"          : jo
        }

experiment = experiments[ args.experiment ]()

time = experiment.simulation[ "time" ]
dt = experiment.simulation[ "dt" ]
x0 = experiment.simulation[ "x0" ]
dxdt = experiment.functions[ "dxdt" ]
V = experiment.functions[ "V" ]
I = experiment.functions[ "I" ]

####

###############################################################################
#                         ODE simulation
###############################################################################

# Plot simulated memristor behaviour
for solv in args.solvers:
    # Solve ODE iteratively using Euler's method
    if solv == "Euler":
        with Timer( title="Euler" ):
            print( "Simulating with Euler solver" )
            x_euler = solver( dxdt, time, dt, x0, method="Euler" )
            x = x_euler
            t = time
            title = "Euler"
    
    # Solve ODE iteratively using Runge-Kutta's method
    if solv == "RK4":
        with Timer( title="RK4" ):
            print( "Simulating with Runge-Kutta solver" )
            x_rk4 = solver( dxdt, time, dt, x0, method="RK4" )
            x = x_rk4
            t = time
            title = "Runge-Kutta"
    
    # Solve ODE with solver
    if solv == "LSODA":
        with Timer( title="LSODA" ):
            print( "Simulating with LSODA solver" )
            x_solve_ivp = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time )
            x = x_solve_ivp.y[ 0, : ]
            t = x_solve_ivp.t
            title = "LSODA"
    
    v = V( t )
    i = I( t, x )
    
    fig1, _, _ = plot_memristor( v, i, t, f"simulated - {title}" )
    fig1.show()
    
    # make video of simulation
    if args.video:
        try:
            os.mkdir( "./videos" )
        except:
            pass
        if not os.path.exists( f"./videos/{experiment.name}_{solv}.mp4" ):
            with Timer( title="Video" ):
                plot_memristor( v, i, t, solv, (10, 4), True, True, f"{experiment.name} - {solv}", True )
            
            ####
    
    ###############################################################################
    #                       Data sampling to solve_ivp solution
    ###############################################################################
    
    # np.random.seed(42)
    noise_percentage = experiment.fitting[ "noise" ]
    
    # Generate noisy data from memristor model
    simulated_data = I( t, x )
    noisy_solution = np.random.normal( simulated_data, np.abs( simulated_data ) * noise_percentage / 100,
                                       size=simulated_data.size )
    
    # Plot noisy data
    fig2, _, _ = plot_memristor( V( t ), noisy_solution, t, f"noisy - {title}" )
    fig2.show()
    
    ####
    
    ###############################################################################
    #                         ODE fitting
    ###############################################################################
    
    # Fit parameters to noisy data
    with Timer( title="curve_fit" ):
        print( "Running curve_fit" )
        popt, pcov = curve_fit( experiment.memristor.fit(), time, noisy_solution,
                                bounds=experiment.fitting[ "bounds" ],
                                p0=experiment.fitting[ "p0" ],
                                # maxfev=100000
                                )
        
        print( "Real parameters ",
               [ (p, v) for p, v in zip( experiment.memristor.parameters(),
                                         experiment.memristor.print_parameters( start='', simple=True ) ) ] )
        print( "Fitted parameters",
               [ (p, np.round( v, 2 )) for p, v in zip( experiment.memristor.parameters(),
                                                        popt ) ] )
    
    # Solve ODE iteratively using Euler's method
    if solv == "Euler":
        with Timer( title="Euler" ):
            print( "Simulating with Euler solver" )
            x_euler_fitted = solver( dxdt, time, dt, x0, method="Euler", args=popt )
            x = x_euler_fitted
            t = time
    
    # Solve ODE iteratively using Runge-Kutta's method
    if solv == "RK4":
        with Timer( title="RK4" ):
            print( "Simulating with Runge-Kutta solver" )
            x_rk4_fitted = solver( dxdt, time, dt, x0, method="RK4", args=popt )
            x = x_rk4_fitted
            t = time
    
    # Solve ODE with solver
    if solv == "LSODA":
        with Timer( title="LSODA" ):
            print( "Simulating with LSODA solver" )
            x_solve_ivp_fitted = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time,
                                            args=popt
                                            )
            x = x_solve_ivp_fitted.y[ 0, : ]
            t = x_solve_ivp_fitted.t
    
    v = V( t )
    i = I( t, x )
    
    # Plot reconstructed data
    fitted_data = I( t, x, *popt )
    fig3, _, _ = plot_memristor( V( t ), fitted_data, t, f"fitted - {title}" )
    fig3.show()
    
    ####
    
    ###############################################################################
    #                         Error
    ###############################################################################
    
    error = np.sum( np.abs( simulated_data[ 1: ] - fitted_data[ 1: ] ) )
    error_average = np.mean( error )
    error_percent = 100 * error / np.sum( np.abs( fitted_data[ 1: ] ) )
    print( f"Average error {order_of_magnitude.symbol( error_average )[ 2 ]}A ({np.mean( error_percent ):.2f} %)" )
    
    ####
    
    ###############################################################################
    #                         Residuals
    ###############################################################################
    
    residuals = noisy_solution - fitted_data
    fig4, axes = plt.subplots( 1, 2, figsize=(10, 4) )
    axes[ 0 ].plot( fitted_data, residuals )
    axes[ 0 ].set_xlabel( "Residuals" )
    axes[ 0 ].set_ylabel( "Fitted values" )
    axes[ 0 ].set_title( "Residuals" )
    stats.probplot( residuals, dist="norm", plot=axes[ 1 ] )
    axes[ 1 ].set_ylabel( "Residuals" )
    axes[ 1 ].set_title( "Residuals" )
    fig4.suptitle( f"Residual analysis - {title}" )
    fig4.tight_layout()
    fig4.show()

####
