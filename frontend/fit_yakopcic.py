import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

from matplotlib import cycler

from backend.functions import *

import matplotlib

mpl.rcParams[ 'axes.prop_cycle' ] = cycler( color=[ "k" ] )
print( matplotlib.pyplot.get_backend() )

###############################################################################
#                         Load data
###############################################################################

with open( f"../pickles/Radius 10 um/-4V_1.pkl", "rb" ) as file:
    df = pickle.load( file )

# remove last datapoint
time = np.array( df[ "t" ].to_list() )[ :-1 ]
current = np.array( df[ "I" ].to_list() )[ :-1 ]
resistance = np.array( df[ "R" ].to_list() )[ :-1 ]
conductance = 1 / resistance
voltage = np.array( df[ "V" ].to_list() )[ :-1 ]

fig_real, _, _ = plot_memristor( df[ "V" ], df[ "I" ], df[ "t" ], "real" )
fig_real.show()

fig1, axes1 = plt.subplots( 2, 1 )
axes1[ 0 ].plot( time, voltage )
axes1[ 1 ].plot( time, conductance )
axes1[ 0 ].set_ylabel( "Voltage (V)" )
axes1[ 1 ].set_ylabel( "Conductivity" )
for ax in np.ravel( axes1 ):
    ax.set_xlabel( "Time (s)" )
fig1.tight_layout()
fig1.show()


################################################
#           Find switching thresholds
################################################

def original_idx( array, mask, fun ):
    idx = np.flatnonzero( mask )
    return idx[ fun( array[ mask ] ) ]


pos_inc_mask = (voltage > 0) & (np.gradient( voltage ) > 0)
neg_dec_mask = (voltage < 0) & (np.gradient( voltage ) < 0)

di_dv = np.gradient( current )

Vp_masked = (np.argmax( di_dv[ pos_inc_mask ] ),
             voltage[ pos_inc_mask ][ np.argmax( di_dv[ pos_inc_mask ] ) ])
Vn_masked = (np.argmin( di_dv[ neg_dec_mask ] ),
             voltage[ neg_dec_mask ][ np.argmin( di_dv[ neg_dec_mask ] ) ])

Vp = (original_idx( di_dv, pos_inc_mask, np.argmax ),
      voltage[ original_idx( di_dv, pos_inc_mask, np.argmax ) ])
Vn = (original_idx( di_dv, neg_dec_mask, np.argmin ),
      voltage[ original_idx( di_dv, neg_dec_mask, np.argmin ) ])

print( "Vp:", Vp, "Vn:", Vn )

fig2, axes2 = plt.subplots( 2, 2 )
axes2[ 0, 0 ].plot( voltage[ pos_inc_mask ], current[ pos_inc_mask ] )
axes2[ 0, 1 ].plot( voltage[ neg_dec_mask ], current[ neg_dec_mask ] )
axes2[ 1, 0 ].plot( voltage[ pos_inc_mask ], di_dv[ pos_inc_mask ] )
axes2[ 1, 1 ].plot( voltage[ neg_dec_mask ], di_dv[ neg_dec_mask ] )
axes2[ 1, 0 ].scatter( voltage[ Vp[ 0 ] ], di_dv[ Vp[ 0 ] ],
                       facecolors='none', edgecolors='r' )
axes2[ 1, 1 ].scatter( voltage[ Vn[ 0 ] ], di_dv[ Vn[ 0 ] ],
                       facecolors='none', edgecolors='r' )
axes2[ 1, 0 ].annotate( r"$V_p$", xy=(voltage[ Vp[ 0 ] ], di_dv[ Vp[ 0 ] ]), color="r" )
axes2[ 1, 1 ].annotate( r"$V_n$", xy=(voltage[ Vn[ 0 ] ], di_dv[ Vn[ 0 ] ]), color="r" )
axes2[ 0, 0 ].set_ylabel( "Current" )
axes2[ 0, 1 ].set_ylabel( "Current" )
axes2[ 1, 0 ].set_ylabel( r"$\Delta$i/$\Delta$v" )
axes2[ 1, 1 ].set_ylabel( r"$\Delta$i/$\Delta$v" )
for ax in np.ravel( axes2 ):
    ax.set_xlabel( "Voltage (V)" )
fig2.tight_layout()
fig2.show()


################################################
#           Fit stable on/off states
################################################

def mim_iv( v, g, b ):
    return g * np.sinh( b * v )


def schottky_iv( v, g, b ):
    return g * np.exp( b * v )


def quadratic_iv( v, a, b, c ):
    return a * np.power( v, 2 ) + b * v + c


def cubic_iv( v, a, b, c, d ):
    return a * np.power( v, 3 ) + b * np.power( v, 2 ) + c * v + d


def mim_mim_iv( v, gp, bp, gn, bn ):
    return np.piecewise( v, [ v >= 0, v < 0 ],
                         [ lambda v: mim_iv( v, gp, bp ), lambda v: mim_iv( v, gn, bn ) ] )


def mim_mim_quad_iv( v, gp, bp, gn, bn, a, b, c ):
    return np.piecewise( v,
                         [ v >= 0,
                           (v <= 0) & (v >= -2),
                           v <= -2 ],
                         [ lambda v: mim_iv( v, gp, bp ),
                           lambda v: mim_iv( v, gn, bn ),
                           lambda v: quadratic_iv( v, a, b, c ) ]
                         )


def mim_cub_iv( v, gp, bp, a, b, c, d ):
    return np.piecewise( v,
                         [ v >= 0,
                           v < 0 ],
                         [ lambda v: mim_iv( v, gp, bp ),
                           lambda v: cubic_iv( v, a, b, c, d ) ]
                         )


on_fit = mim_mim_iv
off_fit = mim_mim_iv

on_mask = ((voltage > 0) & (np.gradient( voltage ) < 0)) \
          | ((voltage < 0) & (np.gradient( voltage ) < 0)
             & (voltage > Vn[ 1 ])
             )
off_mask = ((voltage < 0) & (np.gradient( voltage ) > 0)) \
           | ((voltage > 0) & (np.gradient( voltage ) > 0)
              & (voltage < Vp[ 1 ])
              )

popt_on, pcov_on = scipy.optimize.curve_fit( on_fit, voltage[ on_mask ], current[ on_mask ] )
popt_off, pcov_off = scipy.optimize.curve_fit( off_fit, voltage[ off_mask ], current[ off_mask ] )

gmax_p, bmax_p, gmax_n, bmax_n = popt_on
gmin_p, bmin_p, gmin_n, bmin_n = popt_off

print( "gmax,p:", gmax_p, "bmax,p:", bmax_p, "gmax,n:", gmax_n, "bmax,n:", bmax_n )
print( "gmin,p:", gmin_p, "bmin,p:", bmin_p, "gmin,n:", gmin_n, "bmin,n:", bmin_n )

fig3, axes3 = plt.subplots( 1, 1 )
axes3.scatter( voltage[ on_mask ],
               current[ on_mask ],
               color="b",
               label=f"On state "
                     f"\n ${gmax_p:2.1e}*sinh({bmax_p:.1f}*v) v \geq 0$"
                     f"\n ${gmax_n:2.1e}*sinh({bmax_n:.1f}*v) v<0$"
               )
axes3.scatter( voltage[ on_mask ],
               on_fit( voltage[ on_mask ], *popt_on ),
               s=1 )
axes3.scatter( voltage[ off_mask ],
               current[ off_mask ],
               color="r",
               label=f"Off state "
                     f"\n ${gmin_p:2.1e}*sinh({bmin_p:.1f}*v) v \geq 0$"
                     f"\n ${gmin_n:2.1e}*sinh({bmin_n:.1f}*v) v<0$"
               )
axes3.scatter( voltage[ off_mask ],
               off_fit( voltage[ off_mask ], *popt_off ),
               s=1 )
axes3.annotate( r"$g_{max,p}$", xy=(voltage[ Vn[ 0 ] ], gmin_n), color="g" )
axes3.set_xlabel( "Voltage (V)" )
axes3.set_ylabel( "Current" )
fig3.legend()
fig3.tight_layout()
fig3.show()

################################################
#            Determine Ap and An
################################################

dg_dt = np.gradient( conductance )

g_pk_p = (Vp[ 0 ], dg_dt[ Vp[ 0 ] ])
g_pk_n = (Vn[ 0 ], dg_dt[ Vn[ 0 ] ])

print( "g_pk,p:", g_pk_p, "g_pk,n:", g_pk_n )

Ap = g_pk_p[ 1 ] / (gmax_p - gmin_p)
An = g_pk_n[ 1 ] / (gmax_n - gmin_n)

print( "Ap:", Ap, "An:", An )

fig4, axes4 = plt.subplots( 2, 1 )
axes4[ 0 ].plot( time, conductance )
axes4[ 1 ].plot( time, dg_dt )
axes4[ 1 ].scatter( time[ g_pk_p[ 0 ] ], g_pk_p[ 1 ], facecolors='none', edgecolors='r' )
axes4[ 1 ].scatter( time[ g_pk_n[ 0 ] ], g_pk_n[ 1 ], facecolors='none', edgecolors='r' )
axes4[ 1 ].annotate( r"$g_{pk,p}$", xy=(time[ g_pk_p[ 0 ] ], g_pk_p[ 1 ]), color="r" )
axes4[ 1 ].annotate( r"$g_{pk,n}$", xy=(time[ g_pk_n[ 0 ] ], g_pk_n[ 1 ]), color="r" )
for ax in np.ravel( axes4 ):
    ax.set_xlabel( "Time (s)" )
axes4[ 0 ].set_ylabel( "Conductivity" )
axes4[ 1 ].set_ylabel( r"$\Delta$g/$\Delta$t" )
fig4.tight_layout()
fig4.show()

################################################
#            Determine xp and xn
################################################

g_slow_p = (g_pk_p[ 0 ] + 1, conductance[ g_pk_p[ 0 ] + 1 ])
g_slow_n = (g_pk_n[ 0 ] + 1, conductance[ g_pk_n[ 0 ] + 1 ])

print( "g_slow,p:", g_slow_p, "g_slow_n:", g_slow_n )

xp = (g_slow_p[ 1 ] - gmin_p) / (gmax_p - gmin_p)
xn = (g_slow_n[ 1 ] - gmin_n) / (gmax_n - gmin_n)

print( "xp:", xp, "xn:", xn )

fig5, axes5 = plt.subplots()
axes5.plot( time, conductance )
axes5.scatter( time[ g_slow_p[ 0 ] ], g_slow_p[ 1 ], facecolors='none', edgecolors='r' )
axes5.scatter( time[ g_slow_n[ 0 ] ], g_slow_n[ 1 ], facecolors='none', edgecolors='r' )
axes5.annotate( r"$g_{slow,p}$", xy=(time[ g_slow_p[ 0 ] ], g_slow_p[ 1 ]), color="r" )
axes5.annotate( r"$g_{slow,n}$", xy=(time[ g_slow_n[ 0 ] ], g_slow_n[ 1 ]), color="r" )
axes5.set_xlabel( "Time (s)" )
axes5.set_ylabel( "Conductivity" )
fig5.show()
