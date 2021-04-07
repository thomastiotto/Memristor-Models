import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from block_timer.timer import Timer
import scipy.signal
from scipy.integrate import odeint, solve_ivp
from progressbar import progressbar

##### MATLAB EXAMPLE
eng = matlab.engine.start_matlab()
tf = eng.isprime( 37 )
print( tf )


######

class input_voltage():
    def __init__( self, input_shape, t_max ):
        assert input_shape == "sine" or input_shape == "triangle"
        self.input_shape = input_shape
        self.t_max = t_max
    
    def V( self, t ):
        if self.input_shape == "sine":
            v = np.sin( 2 * np.multiply( np.pi, t ) )
        if self.input_shape == "triangle":
            v = np.abs( scipy.signal.sawtooth( np.pi * t + np.pi / 2, 1 / 2 ) )
            if isinstance( t, np.ndarray ) and len( t ) > 1:
                v[ len( v ) // 2: ] *= -1
            elif t > self.t_max / 2:
                v *= -1
        
        return v


def no_window( **kwargs ):
    return 1


def joglekar( **kwargs ):
    x = kwargs[ "x" ]
    p = kwargs[ "p" ] if "p" in kwargs else 10
    
    return 1 - np.power( np.multiply( 2, x ) - 1, 2 * p )


def biolek( **kwargs ):
    x = kwargs[ "x" ]
    i = kwargs[ "i" ]
    p = kwargs[ "p" ] if "p" in kwargs else 10
    
    return 1 - np.power( x - np.heaviside( -i, 1 ), 2 * p )


class hp_labs():
    def __init__( self, input, window_function, **kwargs ):
        self.V = input.V
        self.F = window_function
        
        self.D = kwargs[ "D" ] if "D" in kwargs else 27e-9
        self.R_ON = kwargs[ "R_ON" ] if "R_ON" in kwargs else 1e2
        self.R_OFF = kwargs[ "R_OFF" ] if "R_OFF" in kwargs else 16e3
        self.R_INIT = kwargs[ "R_INIT" ] if "R_INIT" in kwargs else 11e3
        self.m_D = kwargs[ "m_D" ] if "m_D" in kwargs else 1e-14
    
    def I( self, t, x ):
        return self.V( t ) / (np.multiply( self.R_ON, x ) + np.multiply( self.R_OFF, (np.subtract( 1, x )) ))
    
    def mu_D( self, t, x ):
        i = self.I( t, x )
        
        return ((self.m_D * self.R_ON) / np.power( self.D, 2 )) * i * self.F( x=x, i=i )


t_min = 0
t_max = 8
N = 10000
dt = (t_max - t_min) / N
time = np.arange( t_min, t_max + dt, dt )
# input_shape = "sine"
input_shape = "triangle"
# window_function = no_window
# window_function = joglekar
window_function = biolek

memristor_args = {
        "D"    : 27e-9,
        "R_ON" : 1e2,
        "R_OFF": 16e3,
        "m_D"  : 1e-14,
        }
p = 10
# initial value of state variable
R_INIT = 11e3
x0 = (memristor_args[ "R_OFF" ] - R_INIT) / (memristor_args[ "R_OFF" ] - memristor_args[ "R_ON" ])

x_euler = [ x0 ]
x_rk4 = [ x0 ]
current = [ 0.0 ]
solutions = [ ]

memristor = hp_labs( input_voltage( input_shape, t_max ), window_function, **memristor_args )
dxdt = memristor.mu_D
V = memristor.V
I = memristor.I
F = memristor.F

# Solve ODE iteratively using Euler's method
with Timer( title="Euler" ):
    print( "Running Euler" )
    
    for t in progressbar( time[ :-1 ] ):
        # print(
        #         "Euler", "t", '{:.6e}'.format( t ),
        #         "V", '{:.6e}'.format( V( t ) ),
        #         "I", '{:.6e}'.format( I( t, x_euler[ -1 ] ) ),
        #         "F", '{:.6e}'.format( F( x=x_euler[ -1 ], i=I( t, x_euler[ -1 ] ) ) ),
        #         "x", '{:.6e}'.format( x_euler[ -1 ] ),
        #         "dx", '{:.6e}'.format( dxdt( t, x_euler[ -1 ] ) )
        #         )
        
        current.append( I( t, x_euler[ -1 ] ) )
        
        x_euler.append( x_euler[ -1 ] + dxdt( t, x_euler[ -1 ] ) * dt )
    solutions.append( (x_euler, time, "Euler") )

# Solve ODE iteratively using Runge-Kutta's method
with Timer( title="Runge-Kutta RK4" ):
    print( "Running Runge-Kutta RK4" )
    for t in progressbar( time[ :-1 ], redirect_stdout=True ):
        current.append( I( t, x_rk4[ -1 ] ) )
        
        k1 = dxdt( t, x_rk4[ -1 ] )
        k2 = dxdt( t + dt / 2, x_rk4[ -1 ] + dt * k1 / 2 )
        k3 = dxdt( t + dt / 2, x_rk4[ -1 ] + dt * k2 / 2 )
        k4 = dxdt( t + dt, x_rk4[ -1 ] + dt * k3 )
        x_rk4.append( x_rk4[ -1 ] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6 )
    solutions.append( (x_rk4, time, "Runge-Kutta") )

# Solve ODE with solver
with Timer( title="solve_ivp" ):
    print( "Running solve_ivp" )
    x_solve_ivp = solve_ivp( dxdt, (t_min, t_max), [ x0 ], method="LSODA", t_eval=time )
    solutions.append( (x_solve_ivp.y[ 0, : ], x_solve_ivp.t, "solve_ivp") )

for x, t, title in solutions:
    fig, axes = plt.subplots( 1, 2 )
    ax11 = axes[ 0 ]
    ax11.plot( t, I( t, x ) * 1e6, color="b" )
    ax11.set_ylabel( r'Current ($\mu A$)', color='b' )
    # ax11.set_ylim( [ -150, 150 ] )
    ax11.tick_params( 'y', colors='b' )
    ax12 = ax11.twinx()
    ax12.plot( t, V( t ), color="r" )
    ax12.set_ylabel( 'Voltage (V)', color='r' )
    ax12.tick_params( 'y', colors='r' )
    ax12.set_ylim( [ -1.5, 1.5 ] )
    ax2 = axes[ 1 ]
    ax2.plot( V( t ), I( t, x ) )
    fig.suptitle( f"Memristor Voltage and Current vs. Time ({title})" )
    fig.tight_layout()
    fig.show()
