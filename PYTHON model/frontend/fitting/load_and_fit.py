import os
import pickle
from block_timer.timer import Timer

from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import scipy.stats as stats

from backend.functions import *
from backend.models import Yakopcic

###############################################################################
#                         Load data
###############################################################################

with open( f"../imported_data/pickles/Radius 10 um/-2V_0.pkl", "rb" ) as file:
    df = pickle.load( file )

time = df[ "t" ].to_list()
real_data = df[ "I" ].to_list()
input_voltage = df[ "V" ].to_list()

fig_real, _, _ = plot_memristor( df[ "V" ], df[ "I" ], df[ "t" ], "real" )
fig_real.show()

###############################################################################
#                         ODE fitting
###############################################################################

x0 = 0.1
memristor = Yakopcic( input=Interpolated( x=time, y=input_voltage ), x0=x0 )
dxdt = memristor.dxdt
V = memristor.V
I = memristor.I

# Fit parameters to real data
with Timer( title="curve_fit" ):
    print( "Running curve_fit" )
    popt, pcov = curve_fit( memristor.fit(), time, real_data,
                            bounds=([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 1, 1, 1, 10, 10, 1, 1, 10, 10, 1, 1 ]),
                            p0=[ 0.11, 0.11, 0.5, 7.5, 2, 0.5, 0.75, 1, 5, 0.3, 0.5 ]
                            # p0=[ 0.1, 0.1, 0.1, 1000, 1000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ],
                            # maxfev=100000
                            )
    
    fitted_params = { p: v for p, v in zip( Yakopcic.parameters(), popt ) }
    print( "Fitted parameters", [ (p, np.round( v, 2 )) for p, v in zip( Yakopcic.parameters(), popt ) ] )
    
    # Simulate memristor with fitted parameters
    with Timer( title="solve_ivp" ):
        print( "Running solve_ivp" )
        x_solve_ivp_fitted = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time, args=popt )
    
    # Plot reconstructed data
    fitted_data = I( x_solve_ivp_fitted.t, x_solve_ivp_fitted.y[ 0, : ],
                     fitted_params[ "a1" ], fitted_params[ "a2" ], fitted_params[ "b" ] )
    fig_fitted, _, _ = plot_memristor( V( x_solve_ivp_fitted.t ), fitted_data, x_solve_ivp_fitted.t, "fitted" )
    fig_fitted.show()
    
    ####
    
    ###############################################################################
    #                         Error
    ###############################################################################
    
    error = np.sum( np.abs( real_data[ 1: ] - fitted_data[ 1: ] ) )
    error_average = np.mean( error )
    error_percent = 100 * error / np.sum( np.abs( fitted_data[ 1: ] ) )
    print( f"Average error {order_of_magnitude.symbol( error_average )[ 2 ]}A ({np.mean( error_percent ):.2f} %)" )
    
    ####
    
    ###############################################################################
    #                         Residuals
    ###############################################################################
    
    residuals = real_data - fitted_data
    fig_residuals, axes = plt.subplots( 1, 2, figsize=(10, 4) )
    axes[ 0 ].plot( fitted_data, residuals )
    axes[ 0 ].set_xlabel( "Residuals" )
    axes[ 0 ].set_ylabel( "Fitted values" )
    axes[ 0 ].set_title( "Residuals" )
    stats.probplot( residuals, dist="norm", plot=axes[ 1 ] )
    axes[ 1 ].set_ylabel( "Residuals" )
    axes[ 1 ].set_title( "Residuals" )
    fig_residuals.suptitle( f"Residual analysis" )
    fig_residuals.tight_layout()
    # fig_residuals.show()
    
    ####
