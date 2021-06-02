import pickle

import matplotlib.pyplot as plt
import numpy as np
from block_timer.timer import Timer
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from progressbar import progressbar
import scipy.stats as stats
from order_of_magnitude import order_of_magnitude
import os
import multiprocessing as mp
import argparse

from functions import *
from models import *
from experiments import *

###############################################################################
#                                  Setup
###############################################################################

with open( f"./plots/Radius 10 um/-4V_1.pkl", "rb" ) as file:
    df = pickle.load( file )

# remove last datapoint
time = np.array( df[ "t" ].to_list() )[ :-1 ]
current = np.array( df[ "I" ].to_list() )[ :-1 ]
resistance = np.array( df[ "R" ].to_list() )[ :-1 ]
conductance = 1 / resistance
voltage = np.array( df[ "V" ].to_list() )[ :-1 ]

x0 = 0.11

fig_real, _, _ = plot_memristor( df[ "V" ], df[ "I" ], df[ "t" ], "real" )
fig_real.show()


###############################################################################
#                         ODE fitting
###############################################################################

def g( v, Ap, An, Vp, Vn ):
    if v > Vp:
        return Ap * (np.exp( v ) - np.exp( Vp ))
    elif v < -Vn:
        return -An * (np.exp( -v ) - np.exp( Vn ))
    else:
        return 0


def wp( x, xp ):
    return (xp - x) / (1 - xp) + 1


def wn( x, xn ):
    return x / (1 - xn)


def f( v, x, xp, xn, eta ):
    if eta * v >= 0:
        if x >= xp:
            return np.exp( -(x - xp) ) * wp( x, xp )
        else:
            return 1
    else:
        if x <= 1 - xn:
            return np.exp( (x + xn - 1) ) * wn( x, xn )
        else:
            return 1


V = Interpolated( time, voltage )


def dxdt( t, x, Ap, An, Vp, Vn, xp, xn ):
    eta = 1
    v = V( t )
    return eta * g( v, Ap, An, Vp, Vn ) * f( v, x, xp, xn, eta )


def ohmic_iv( v, g ):
    return g * v


def mim_iv( v, g, b ):
    return g * np.sinh( b * v )


def mim_mim_iv( v, gp, bp, gn, bn ):
    return np.piecewise( v, [ v < 0, v >= 0 ],
                         [ lambda v: mim_iv( v, gn, bn ), lambda v: mim_iv( v, gp, bp ) ] )


def I_mim_mim( t, x, gmax, bmax, gmin, bmin ):
    v = V( t )
    return mim_iv( v, gmax, bmax ) * x + mim_iv( v, gmin, bmin ) * (1 - x)


def I_mim_mim_mim_mim( t, x, gmaxp, bmaxp, gmaxn, bmaxn, gminp, bminp, gminn, bminn ):
    v = V( t )
    return mim_mim_iv( v, gmaxp, bmaxp, gmaxn, bmaxn ) * x + mim_mim_iv( v, gminp, bminp, gminn, bminn ) * (1 - x)


I = I_mim_mim_mim_mim

on_par = { "gmaxp": 9.89356358e-05,
           "bmaxp": 4.95768018e+00,
           "gmaxn": 1.38215701e-05,
           "bmaxn": 3.01625878e+00 }
off_par = { "gminp": 1.21787202e-05,
            "bminp": 7.10131146e+00,
            "gminn": 4.36419052e-07,
            "bminn": 2.59501160e+00
            }


def ode_fitting( t, Ap, An, Vp, Vn, xp, xn ):
    print( Ap, An, Vp, Vn, xp, xn )
    sol = solve_ivp( dxdt, (t[ 0 ], t[ -1 ]), [ x0 ], method="LSODA",
                     t_eval=t,
                     args=[ Ap, An, Vp, Vn, xp, xn ],
                     )
    
    return I( t, sol.y[ 0, : ], **on_par, **off_par )


bounds = {
        "Ap": 1e4,
        "An": 1e4,
        "Vp": 1,
        "Vn": 1,
        "xp": 1,
        "xn": 1
        }
p0 = {
        "Ap": 1,
        "An": 1,
        "Vp": 0,
        "Vn": 0,
        "xp": 0.1,
        "xn": 0.1
        }
# Fit parameters to data
with Timer( title="curve_fit" ):
    print( "Running curve_fit" )
    popt, pcov = curve_fit( ode_fitting, time, current,
                            bounds=(np.zeros( len( bounds ) ), list( bounds.values() )),
                            p0=list( p0.values() ),
                            maxfev=100000
                            )
    
    print( "Fitted parameters", [ (k, p) for k, p in zip( bounds.keys(), popt ) ] )
    
    # Simulate memristor model with fitted parameters
    with Timer( title="solve_ivp" ):
        print( "Running solve_ivp" )
        x_solve_ivp_fitted = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time,
                                        args=popt
                                        )
    
    # Plot reconstructed data
    fitted_data = I( x_solve_ivp_fitted.t, x_solve_ivp_fitted.y[ 0, : ], **on_par, **off_par )
    fig_fitted, _, _ = plot_memristor( V( x_solve_ivp_fitted.t ), fitted_data, x_solve_ivp_fitted.t, "fitted" )
    fig_fitted.show()
    
    ####
    
    ###############################################################################
    #                         Error
    ###############################################################################
    
    error = np.sum( np.abs( current[ 1: ] - fitted_data[ 1: ] ) )
    error_average = np.mean( error )
    error_percent = 100 * error / np.sum( np.abs( fitted_data[ 1: ] ) )
    print( f"Average error {order_of_magnitude.symbol( error_average )[ 2 ]}A ({np.mean( error_percent ):.2f} %)" )
    
    ####
    
    ###############################################################################
    #                         Residuals
    ###############################################################################
    
    residuals = current - fitted_data
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
