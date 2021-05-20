import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle

from scipy.optimize import curve_fit

from functions import *
from models import Yakopcic

###############################################################################
#                         Load data
###############################################################################

with open( f"./plots/Radius 10 um/-4V_1.pkl", "rb" ) as file:
    df = pickle.load( file )

time = df[ "t" ].to_list()
real_data = df[ "I" ].to_list()
input_voltage = df[ "V" ].to_list()

fig, _, _ = plot_memristor( df[ "V" ], df[ "I" ], df[ "t" ], "Test" )
fig.show()

###############################################################################
#                         ODE fitting
###############################################################################


memristor = Yakopcic( input=None )

# Fit parameters to noisy data
with Timer( title="curve_fit" ):
    print( "Running curve_fit" )
    popt, pcov = curve_fit( Yakopcic.fit(), time, real_data,
                            # bounds=experiment.fitting[ "bounds" ],
                            # p0=experiment.fitting[ "p0" ],
                            # maxfev=100000
                            )
    
    print( "Fitted parameters",
           [ (p, np.round( v, 2 )) for p, v in zip( Yakopcic.parameters(),
                                                    popt ) ] )
    
    # Simulate memristor with fitted parameters
    with Timer( title="solve_ivp" ):
        print( "Running solve_ivp" )
        x_solve_ivp_fitted = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time,
                                        args=popt
                                        )
    
    # Plot reconstructed data
    fitted_data = I( x_solve_ivp_fitted.t, x_solve_ivp_fitted.y[ 0, : ], *popt )
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
