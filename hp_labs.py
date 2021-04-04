import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from block_timer.timer import Timer
import scipy.signal
from scipy.integrate import odeint, solve_ivp

##### MATLAB EXAMPLE
eng = matlab.engine.start_matlab()
tf = eng.isprime( 37 )
print( tf )


######

class input_voltage():
    def __init__( self, input_shape ):
        assert input_shape == "sine" or input_shape == "triangle"
        self.input_shape = input_shape
    
    def V( self, t ):
        if self.input_shape == "sine":
            v = np.sin( 2 * np.multiply( np.pi, t ) )
        if self.input_shape == "triangle":
            v = scipy.signal.sawtooth( np.pi * t + np.pi / 2, 1 / 2 )
            if isinstance( t, list ):
                v[ len( v ) // 2: ] = -1 * v[ len( v ) // 2: ]
        
        return v


class hp_labs():
    def __init__( self, input ):
        self.V = input.V
    
    def I( self, t, x ):
        return self.V( t ) / (np.multiply( R_ON, x ) + np.multiply( R_OFF, (np.subtract( 1, x )) ))
    
    def mu_D( self, t, x ):
        return ((m_D * R_ON) / np.power( D, 2 )) * self.I( t, x ) * self.F( x )
    
    def F( self, x ):
        return (1 - np.power( np.multiply( 2, x ) - 1, 2 * p ))


tmin = 0
tmax = 2
N = 1000
dt = (tmax - tmin) / N
time = np.arange( tmin, tmax + dt, dt )
input_shape = "triangle"

D = 10e-9
R_ON = 1e2
R_OFF = 16e3
R_INIT = 11e3
m_D = 1e-14
p = 10
x0 = (R_OFF - R_INIT) / (R_OFF - R_ON)

x_euler = [ x0 ]
x_rk4 = [ x0 ]
current = [ 0.0 ]

memristor = hp_labs( input_voltage( "sine" ) )
dxdt = memristor.mu_D
V = memristor.V
I = memristor.I

# Solve ODE
with Timer( title="odeint" ):
    x_odeint = odeint( dxdt, x0, time )
with Timer( title="solve_ivp" ):
    x_solve_ivp = solve_ivp( dxdt, (tmin, tmax), [ x0 ], method="LSODA", t_eval=time )

# Solve ODE iteratively using Euler's method
with Timer( title="Euler" ):
    for t in time[ :-1 ]:
        current.append( I( t, x_euler[ -1 ] ) )
        x_euler.append( x_euler[ -1 ] + memristor.mu_D( t, x_euler[ -1 ] ) * dt )
        
        # print( "Euler", "t", '{:.6e}'.format( t ), "I", '{:.6e}'.format( I( t, x_euler[ -1 ] ) ), "dx",
        #        '{:.6e}'.format( v_D( t, x_euler[ -1 ] ) ) )

# Solve ODE iteratively using Runge-Kutta's method
with Timer( title="Runge-Kutta" ):
    for t in time[ :-1 ]:
        current.append( I( t, x_rk4[ -1 ] ) )
        
        k1 = memristor.mu_D( t, x_rk4[ -1 ] )
        k2 = memristor.mu_D( t + dt / 2, x_rk4[ -1 ] + dt * k1 / 2 )
        k3 = memristor.mu_D( t + dt / 2, x_rk4[ -1 ] + dt * k2 / 2 )
        k4 = memristor.mu_D( t + dt, x_rk4[ -1 ] + dt * k3 )
        x_rk4.append( x_rk4[ -1 ] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6 )
        
        # print( "RK4", "t", '{:.6e}'.format( t ), "I", '{:.6e}'.format( I( t, x_rk4[ -1 ] ) ), "dx",
        #        '{:.6e}'.format( v_D( t, x_rk4[ -1 ] ) ) )

for x, t, title in zip(
        [ x_euler, x_rk4, x_odeint[ :, 0 ], x_solve_ivp.y[ 0, : ] ],
        [ time, time, time, x_solve_ivp.t ],
        [ "Euler", "RK4", "odeint", "solve_ivp" ]
        ):
    fig, axes = plt.subplots( 1, 2 )
    ax11 = axes[ 0 ]
    ax11.plot( t, I( t, x ) * 1e6, color="b" )
    ax11.set_ylabel( r'Current ($\mu A$)', color='b' )
    ax11.set_ylim( [ -150, 150 ] )
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
