import pickle
from scipy.integrate import solve_ivp

from matplotlib.widgets import Slider, Button

from backend.functions import *
from backend.experiments import *

experiment = oblea_sine()

time = experiment.simulation[ "time" ]
x0 = experiment.simulation[ "x0" ]
dxdt = experiment.functions[ "dxdt" ]
V = experiment.functions[ "V" ]
I = experiment.functions[ "I" ]

## Initial plot
x_solve_ivp = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time )

t = x_solve_ivp.t
x = x_solve_ivp.y[ 0, : ]

v = V( t )
i = I( t, x )

fig, lines, axes = plot_memristor( v, i, t, "Yakopcic", figsize=(12, 6), iv_arrows=False )

################################################
#                       GUI
################################################

colour = "lightgoldenrodyellow"

# buttons
ax_switchV = plt.axes( [ 0.005, 0.22, 0.1, 0.04 ] )
button_input = Button( ax_switchV, "Switch V", color=colour, hovercolor='0.975' )

ax_reset = plt.axes( [ 0.7, 0.85, 0.1, 0.04 ] )
button_reset = Button( ax_reset, 'Reset', color=colour, hovercolor='0.975' )

ax_load = plt.axes( [ 0.81, 0.85, 0.1, 0.04 ] )
button_load = Button( ax_load, 'Load data', color=colour, hovercolor='0.975' )

# create the voltage sliders
fig.subplots_adjust( left=0.15 )
voltage_sliders = [ ]
ax_vp = plt.axes( [ 0.02, 0.3, 0.02, 0.25 ], facecolor=colour )
slider_vp = Slider(
        ax_vp,
        r"$V^+$",
        valmin=0,
        valmax=4,
        valinit=experiment.input_args[ "vp" ],
        valstep=0.1,
        valfmt=r"%.2f $V$",
        orientation="vertical"
        )
voltage_sliders.append( slider_vp )
ax_vn = plt.axes( [ 0.07, 0.3, 0.02, 0.25 ], facecolor=colour )
slider_vn = Slider(
        ax_vn,
        r"$V^-$",
        valmin=0,
        valmax=10,
        valinit=experiment.input_args[ "vn" ],
        valstep=0.1,
        valfmt=r"%.2f $V$",
        orientation="vertical"
        )
voltage_sliders.append( slider_vn )
ax_f = plt.axes( [ 0.045, 0.65, 0.02, 0.25 ], facecolor=colour )
slider_frequency = Slider(
        ax_f,
        r"$\nu$",
        valmin=0.05,
        valmax=100,
        valinit=experiment.input_args[ "frequency" ],
        valstep=0.1,
        valfmt=r"%.2f $Hz$",
        orientation="vertical"
        )
voltage_sliders.append( slider_frequency )

# create experiment sliders
fig.subplots_adjust( top=0.8 )
experiment_sliders = [ ]
ax_time = plt.axes( [ 0.15, 0.85, 0.36, 0.03 ], facecolor=colour )
slider_time = Slider(
        ax_time,
        r"Time",
        valmin=1e-3,
        valmax=80,
        valstep=10e-3,
        closedmin=False,
        valinit=experiment.simulation[ "t_max" ],
        valfmt=r"%.0f $s$"
        )
experiment_sliders.append( slider_time )

# create the memristor sliders
fig.subplots_adjust( bottom=0.3 )
memristor_sliders = [ ]
## I parameters
ax_a1 = plt.axes( [ 0.05, 0.15, 0.15, 0.03 ], facecolor=colour )
slider_a1 = Slider(
        ax_a1,
        r"$a_1$",
        valmin=0,
        valmax=2,
        valinit=experiment.memristor.a1,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_a1 )
ax_a2 = plt.axes( [ 0.05, 0.1, 0.15, 0.03 ], facecolor=colour )
slider_a2 = Slider(
        ax_a2,
        r"$a_2$",
        valmin=0,
        valmax=2,
        valinit=experiment.memristor.a2,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_a2 )
ax_b = plt.axes( [ 0.05, 0.05, 0.15, 0.03 ], facecolor=colour )
slider_b = Slider(
        ax_b,
        r"$b$",
        valmin=0,
        valmax=1,
        valinit=experiment.memristor.b,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_b )

## g parameters
ax_Ap = plt.axes( [ 0.3, 0.17, 0.15, 0.03 ], facecolor=colour )
slider_Ap = Slider(
        ax_Ap,
        r"$A_p$",
        valmin=0,
        valmax=1e10,
        valinit=experiment.memristor.Ap,
        valfmt=r"%.2E"
        )
memristor_sliders.append( slider_Ap )
ax_An = plt.axes( [ 0.3, 0.12, 0.15, 0.03 ], facecolor=colour )
slider_An = Slider(
        ax_An,
        r"$A_n$",
        valmin=0,
        valmax=1e10,
        valinit=experiment.memristor.An,
        valfmt=r"%.2E"
        )
memristor_sliders.append( slider_An )
ax_Vp = plt.axes( [ 0.3, 0.07, 0.15, 0.03 ], facecolor=colour )
slider_Vp = Slider(
        ax_Vp,
        r"$V_p$",
        valmin=0,
        valmax=4,
        valinit=experiment.memristor.Vp,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_Vp )
ax_Vn = plt.axes( [ 0.3, 0.02, 0.15, 0.03 ], facecolor=colour )
slider_Vn = Slider(
        ax_Vn,
        r"$V_n$",
        valmin=0,
        valmax=4,
        valinit=experiment.memristor.Vn,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_Vn )

## f parameters
ax_alphap = plt.axes( [ 0.55, 0.17, 0.15, 0.03 ], facecolor=colour )
slider_alphap = Slider(
        ax_alphap,
        r"$\alpha_p$",
        valmin=0,
        valmax=30,
        valinit=experiment.memristor.alphap,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_alphap )
ax_alphan = plt.axes( [ 0.55, 0.12, 0.15, 0.03 ], facecolor=colour )
slider_alphan = Slider(
        ax_alphan,
        r"$\alpha_n$",
        valmin=0,
        valmax=30,
        valinit=experiment.memristor.alphan,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_alphan )
ax_xp = plt.axes( [ 0.55, 0.07, 0.15, 0.03 ], facecolor=colour )
slider_xp = Slider(
        ax_xp,
        r"$x_p$",
        valmin=0,
        valmax=1,
        valinit=experiment.memristor.xp,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_xp )
ax_xn = plt.axes( [ 0.55, 0.02, 0.15, 0.03 ], facecolor=colour )
slider_xn = Slider(
        ax_xn,
        r"$x_n$",
        valmin=0,
        valmax=1,
        valinit=experiment.memristor.xn,
        valfmt=r"%.2f"
        )
memristor_sliders.append( slider_xn )

## Other parameters
ax_eta = plt.axes( [ 0.8, 0.15, 0.15, 0.03 ], facecolor=colour )
slider_eta = Slider(
        ax_eta,
        r"$\eta$",
        valmin=-1,
        valmax=1,
        valinit=experiment.memristor.eta,
        valstep=1,
        valfmt=r"%.0f"
        )
memristor_sliders.append( slider_eta )

sliders = voltage_sliders + experiment_sliders + memristor_sliders


################################################
#                 Event handlers
################################################

def load_data( event ):
    with open( f"../plots/Radius 10 um/-4V_1.pkl", "rb" ) as file:
        df = pickle.load( file )
    
    time = df[ "t" ].to_list()
    data = df[ "I" ].to_list()
    input_voltage = df[ "V" ].to_list()
    
    switch_input( Interpolated( x=time, y=input_voltage ) )
    
    sim_time = experiment.simulation[ "time" ]
    
    # Plot new graphs
    # axes[ 0 ].plot( sim_time, data[ :len( sim_time ) ], color="g-", alpha=0.5 )
    # axes[ 2 ].plot( input_voltage[ :len( sim_time ) ], data[ :len( sim_time ) ], color="g-", alpha=0.5 )
    
    update( 0 )


def switch_input( input ):
    global V
    
    if not input:
        if isinstance( experiment.input_function, Sine ):
            experiment.input_function = Triangle( **experiment.input_args )
        elif isinstance( experiment.input_function, Triangle ):
            experiment.input_function = Sine( **experiment.input_args )
    else:
        experiment.input_function = input
    
    # update references
    experiment.memristor.V = experiment.input_function.input_function
    experiment.functions[ "V" ] = experiment.memristor.V
    V = experiment.functions[ "V" ]
    
    update( 0 )


def reset( event ):
    for s in sliders:
        s.reset()
    
    update( 0 )


def update( event ):
    # Read updated time from slider
    experiment.recalculate_time( slider_time.val )
    
    # Adjust to new limits
    axes[ 0 ].set_xlim( [ 0, slider_time.val ] )
    axes[ 1 ].set_xlim( [ 0, slider_time.val ] )
    
    time = experiment.simulation[ "time" ]
    
    # Read updated voltage from sliders
    experiment.input_function.vp = slider_vp.val
    experiment.input_function.vn = slider_vn.val
    experiment.input_function.frequency = slider_frequency.val
    
    v = V( time )
    
    # Adjust to new limits
    axes[ 1 ].set_ylim( [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
    axes[ 2 ].set_xlim( [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
    
    # Read updated memristor parameters from sliders
    memristor_args = [ sl.val for sl in memristor_sliders ]
    
    # Simulate memristor with updated values
    x_solve_ivp = solve_ivp( dxdt, (time[ 0 ], time[ -1 ]), [ x0 ], method="LSODA", t_eval=time, args=memristor_args )
    x = x_solve_ivp.y[ 0, : ]
    i = I( time, x, *memristor_args )
    
    i_oom = order_of_magnitude.symbol( np.max( i ) )
    i_scaled = i * 1 / i_oom[ 0 ]
    
    # remove old lines
    axes[ 0 ].lines.pop( 0 )
    axes[ 1 ].lines.pop( 0 )
    axes[ 2 ].lines.pop( 0 )
    
    # Plot new graphs
    axes[ 0 ].plot( experiment.simulation[ "time" ], i_scaled, color="b" )
    axes[ 1 ].plot( experiment.simulation[ "time" ], v, color="r" )
    axes[ 2 ].plot( v, i_scaled, color="b" )
    
    # Adjust to new limits
    axes[ 0 ].set_ylim( [ np.min( i_scaled ) - np.abs( 0.5 * np.min( i_scaled ) ),
                          np.max( i_scaled ) + np.abs( 0.5 * np.max( i_scaled ) ) ] )
    axes[ 0 ].set_ylabel( f"Current ({i_oom[ 1 ]}A)", color="b" )
    axes[ 2 ].set_ylim( [ np.min( i_scaled ) - np.abs( 0.5 * np.min( i_scaled ) ),
                          np.max( i_scaled ) + np.abs( 0.5 * np.max( i_scaled ) ) ] )
    
    fig.canvas.draw()


################################################
#          Event handlers registration
################################################

button_load.on_clicked( load_data )
button_input.on_clicked( switch_input )
button_reset.on_clicked( reset )

for s in sliders:
    s.on_changed( update )

plt.show()

load_data( 0 )
