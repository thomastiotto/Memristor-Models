import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from backend.functions import *

###############################################################################
#                         Load data
###############################################################################

with open( f"../pickles/Radius 10 um/-2V_3.pkl", "rb" ) as file:
    df = pickle.load( file )

time = df[ "t" ].to_list()
current = df[ "I" ].to_list()
voltage = df[ "V" ].to_list()

# self.time = np.array( df[ "t" ].to_list() )[ :-1 ]
# self.current = np.array( df[ "I" ].to_list() )[ :-1 ]
# self.voltage = np.array( df[ "V" ].to_list() )[ :-1 ]

###############################################################################
#                         Plot data
###############################################################################

fig, axes = plt.subplots( 2, 2, figsize=(12, 8) )
axes[ 0, 0 ].plot( time, current, c="b" )
axv1 = axes[ 0, 0 ].twinx()
axv1.plot( time, voltage, c="r" )
axes[ 0, 1 ].plot( voltage, current )

###############################################################################
#                         Define model
###############################################################################

V = Interpolated( time, voltage )


def I( t, x, gmax_p, gmin_p, bmax_p, bmin_p, gmax_n, gmin_n, bmax_n, bmin_n ):
    v = V( t )
    
    i = np.where( v >= 0,
                  gmax_p * np.sinh( bmax_p * v ) * x + gmin_p * np.sinh( bmin_p * v ) * (1 - x),
                  gmax_n * np.sinh( bmax_n * v ) * x + gmin_n * np.sinh( bmin_n * v ) * (1 - x)
                  )
    
    return i


def g( v, Ap, An, Vp, Vn ):
    return np.select( [ v > Vp, v < -Vn ],
                      [ Ap * (np.exp( v ) - np.exp( Vp )),
                        -An * (np.exp( -v ) - np.exp( Vn )) ],
                      default=0 )


def wp( x, xp ):
    return ((xp - x) / (1 - xp)) + 1


def wn( x, xn ):
    return x / xn


def f( v, x, xp, xn, alphap, alphan, eta ):
    return np.select( [ eta * v >= 0, eta * v < 0 ],
                      [ np.select( [ x >= xp, x < xp ],
                                   [ np.exp( -alphap * (x - xp) ) * wp( x, xp ),
                                     1 ] ),
                        np.select( [ x <= xn, x > xn ],
                                   [ np.exp( alphan * (x - xn) ) * wn( x, xn ),
                                     1 ] )
                        ] )


def dxdt( t, x, Ap, An, Vp, Vn, xp, xn, alphap, alphan, eta ):
    v = V( t )
    
    return eta * g( v, Ap, An, Vp, Vn ) * f( v, x, xp, xn, alphap, alphan, eta )


###############################################################################
#                         Define parameters
###############################################################################

Ap = 90
An = 10

Vp = 0.5
Vn = 0.5

xp = 0.1
xn = 0.242

alphap = 1
alphan = 1

eta = 1

gmax_p = 9e-5
bmax_p = 4.96
gmin_p = 1.5e-5
bmin_p = 6.91
gmax_n = 1.7e-4
bmax_n = 3.23
gmin_n = 4.4e-7
bmin_n = 2.6

xo = 0.11

params_dxdt = [ Ap, An, Vp, Vn, xp, xn, alphap, alphan, eta ]
params_I = [ gmax_p, gmin_p, bmax_p, bmin_p, gmax_n, gmin_n, bmax_n, bmin_n ]

###############################################################################
#                         Simulate
###############################################################################


# x_solve_ivp = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]),
#                          [ xo ],
#                          method="LSODA",
#                          t_eval=time,
#                          args=params_dxdt )
#
# time_sim = x_solve_ivp.t
# x = x_solve_ivp.y[ 0, : ]
# voltage_sim = V( time_sim )
# current_sim = I( time_sim, x, *params_I )

x = solver( dxdt, time, 1 / 10000, xo, params_dxdt )
time_sim = time
current_sim = I( time_sim, x, *params_I )
voltage_sim = V( time_sim )

###############################################################################
#                         Plot simulation
###############################################################################

axes[ 1, 0 ].plot( time_sim, np.multiply( current_sim, 1000 ), c="b" )
axv2 = axes[ 1, 0 ].twinx()
axv2.plot( time_sim, voltage_sim, c="r" )
axes[ 1, 1 ].plot( voltage_sim, np.multiply( current_sim, 1000 ) )
for ax in axes.flatten():
    ax.set_xlabel( "Time" )
    ax.set_ylabel( "Current (mA)" )
fig.show()

plt.figure()
plt.plot( voltage, np.multiply( current, 1000 ), c="r" )
plt.plot( voltage_sim, np.multiply( current_sim, 1000 ), c="k" )
plt.xlabel( "Voltage (V)" )
plt.ylabel( "Current (mA)" )
plt.show()

plt.figure()
plt.plot( time_sim, x )
plt.show()
