import matplotlib.pyplot as plt
from block_timer.timer import Timer
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from progressbar import progressbar
import scipy.stats as stats
from order_of_magnitude import order_of_magnitude
import os
import multiprocessing as mp

from functions import *
from models import *
from experiments import *

###############################################################################
#                                  Setup
###############################################################################

experiment = hp_labs_pulsed()

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
solver = "LSODA"

solutions = [ ]

# # Solve ODE iteratively using Euler's method
# x_euler = euler_solver(dxdt, time, dt, x0)
# solutions.append((x_euler, time, "Euler"))

# # Solve ODE iteratively using Runge-Kutta's method
# x_rk4 = rk4_solver(dxdt, time, dt, x0)
# solutions.append((x_rk4, time, "Runge-Kutta"))

# Solve ODE with solver
with Timer( title="solve_ivp" ):
    print( "Running solve_ivp" )
    x_solve_ivp = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method=solver, t_eval=time )
    solutions.append( (x_solve_ivp.y[ 0, : ], x_solve_ivp.t, "solve_ivp") )

# Plot simulated memristor behaviour
for x, t, title in solutions:
    v = V( t )
    i = I( t, x )
    
    fig, _, _ = plot_memristor( v, i, t, "simulated" )
    fig.show()
    
    # make video of simulation
    if not os.path.exists( f"{experiment.name}.mp4" ):
        if __name__ == "__main__":
            mp.set_start_method( "fork" )
            p = mp.Process( target=plot_memristor,
                            args=(v, i, t, title, solver, True, experiment.name) )
            p.start()
####


###############################################################################
#                       Data sampling to solve_ivp solution
###############################################################################

np.random.seed( 42 )
noise_percentage = experiment.fitting[ "noise" ]

# Generate noisy data from memristor model
simulated_data = I( x_solve_ivp.t, x_solve_ivp.y[ 0, : ] )
noisy_solution = np.random.normal( simulated_data, np.abs( simulated_data ) * noise_percentage / 100,
                                   size=simulated_data.size )

# Plot noisy data
fig2, _, _ = plot_memristor( V( x_solve_ivp.t ), noisy_solution, x_solve_ivp.t, "noisy" )
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
                            # p0=[10e-9, 10e3, 100e3, 1e-14]
                            )

print( "curve_fit parameters", end=" " )
experiment.memristor.print_parameters( start="", simple=True )
print( "Fitted parameters", popt )

# Simulate memristor with fitted parameters
with Timer( title="solve_ivp" ):
    print( "Running solve_ivp" )
    x_solve_ivp_fitted = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time,
                                    args=popt
                                    )

# Plot reconstructed data
fitted_data = I( x_solve_ivp_fitted.t, x_solve_ivp_fitted.y[ 0, : ] )
fig3, _, _ = plot_memristor( V( x_solve_ivp_fitted.t ), fitted_data, x_solve_ivp_fitted.t, "fitted" )
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
fig4.suptitle( f"Residual analysis" )
fig4.tight_layout()
fig4.show()

####
