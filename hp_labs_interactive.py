import copy

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import ScalarFormatter

import functions
from functions import *
from models import *
from experiments import *

# TODO vary input voltage
# TODO switch input voltage type
experiment = hp_labs_sine()

time = experiment.simulation[ "time" ]
dt = experiment.simulation[ "dt" ]
x0 = experiment.simulation[ "x0" ]
dxdt = experiment.functions[ "dxdt" ]
V = experiment.functions[ "V" ]
I = experiment.functions[ "I" ]

x_solve_ivp = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time )

t = x_solve_ivp.t
x = x_solve_ivp.y[ 0, : ]

v = V( t )
i = I( t, x )

fig, lines, axes = plot_memristor( v, i, t, experiment.name, figsize=(12, 6), iv_arrows=False )

colour = "lightgoldenrodyellow"
# create the voltage sliders
fig.subplots_adjust( left=0.15 )
voltage_sliders = [ ]
ax_vp = plt.axes( [ 0.02, 0.3, 0.02, 0.25 ], facecolor=colour )
svp = Slider(
        ax_vp,
        r"$V^+$",
        valmin=0,
        valmax=10,
        valinit=experiment.input_args[ "vp" ],
        valstep=0.1,
        valfmt=r"%.2f $V$",
        orientation="vertical"
        )
voltage_sliders.append( svp )
ax_vn = plt.axes( [ 0.07, 0.3, 0.02, 0.25 ], facecolor=colour )
svn = Slider(
        ax_vn,
        r"$V^-$",
        valmin=0,
        valmax=10,
        valinit=experiment.input_args[ "vn" ],
        valstep=0.1,
        valfmt=r"%.2f $V$",
        orientation="vertical"
        )
voltage_sliders.append( svn )
ax_f = plt.axes( [ 0.045, 0.65, 0.02, 0.25 ], facecolor=colour )
sf = Slider(
        ax_f,
        r"$\nu$",
        valmin=0,
        valmax=100,
        valinit=experiment.input_args[ "frequency" ],
        valstep=1,
        valfmt=r"%.2f $Hz$",
        orientation="vertical"
        )
voltage_sliders.append( sf )

# create the memristor sliders
fig.subplots_adjust( bottom=0.3 )
memristor_sliders = [ ]
ax_d = plt.axes( [ 0.05, 0.15, 0.25, 0.03 ], facecolor=colour )
sd = Slider(
        ax_d,
        r"$D$",
        valmin=1e-9,
        valmax=100e-9,
        valinit=experiment.memristor.D,
        valfmt=r"%.2E $m$"
        )
memristor_sliders.append( sd )
ax_ron = plt.axes( [ 0.05, 0.1, 0.25, 0.03 ], facecolor=colour )
sron = Slider(
        ax_ron,
        r"$R_{ON}$",
        valmin=1e3,
        valmax=100e3,
        valinit=experiment.memristor.RON,
        valfmt=r"%.2E $\Omega$"
        )
memristor_sliders.append( sron )
ax_roff = plt.axes( [ 0.5, 0.15, 0.25, 0.03 ], facecolor=colour )
sroff = Slider(
        ax_roff,
        r"$R_{OFF}$",
        valmin=10e3,
        valmax=1000e3,
        valinit=experiment.memristor.ROFF,
        valfmt=r"%.2E $\Omega$"
        )
memristor_sliders.append( sroff )
ax_mud = plt.axes( [ 0.5, 0.1, 0.25, 0.03 ], facecolor=colour )
smud = Slider(
        ax_mud,
        r"$\mu_D$",
        valmin=1e-15,
        valmax=10e-14,
        valinit=experiment.memristor.muD,
        valfmt=r"%.2E $m^2 s^{-1} V^{-1}$"
        )
memristor_sliders.append( smud )


def update_voltage( val ):
    memristor_args = args = [ sl.val for sl in memristor_sliders ]
    experiment.input_function.vp = svp.val
    experiment.input_function.vn = svn.val
    experiment.input_function.frequency = sf.val
    
    x_solve_ivp = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=t, args=memristor_args )
    x = x_solve_ivp.y[ 0, : ]
    
    i = I( t, x )
    v = V( t )
    
    # update voltage
    axes[ 1 ].set_ylim( [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
    lines[ 1 ].set_ydata( v )
    
    # update memristor
    axes[ 0 ].set_ylim( [ np.min( i ) - np.abs( 0.5 * np.min( i ) ), np.max( i ) + np.abs( 0.5 * np.max( i ) ) ] )
    axes[ 2 ].set_ylim( [ np.min( i ) - np.abs( 0.5 * np.min( i ) ), np.max( i ) + np.abs( 0.5 * np.max( i ) ) ] )
    lines[ 0 ].set_ydata( i )
    lines[ 2 ].set_ydata( i )
    
    fig.canvas.draw_idle()


def update_memristor( val ):
    memristor_args = [ sl.val for sl in memristor_sliders ]
    
    x_solve_ivp = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=t, args=memristor_args )
    x = x_solve_ivp.y[ 0, : ]
    i = I( t, x )
    
    axes[ 0 ].set_ylim( [ np.min( i ) - np.abs( 0.5 * np.min( i ) ), np.max( i ) + np.abs( 0.5 * np.max( i ) ) ] )
    axes[ 2 ].set_ylim( [ np.min( i ) - np.abs( 0.5 * np.min( i ) ), np.max( i ) + np.abs( 0.5 * np.max( i ) ) ] )
    lines[ 0 ].set_ydata( i )
    lines[ 2 ].set_ydata( i )
    
    fig.canvas.draw_idle()


for s in voltage_sliders:
    s.on_changed( update_voltage )
for s in memristor_sliders:
    s.on_changed( update_memristor )

ax_reset = plt.axes( [ 0.35, 0.025, 0.1, 0.04 ] )
button = Button( ax_reset, 'Reset', color=colour, hovercolor='0.975' )


def reset( event ):
    for sv, sm in zip( voltage_sliders, memristor_sliders ):
        sv.reset()
        sm.reset()


button.on_clicked( reset )

plt.show()
