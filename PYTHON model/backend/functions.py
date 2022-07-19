import numpy as np
import os
import scipy.signal
from scipy import interpolate
from order_of_magnitude import order_of_magnitude

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from tqdm import tqdm


def ohmic_iv( v, g ):
    return g * v


def mim_iv( v, g, b ):
    return g * np.sinh( b * v )


def schottky_iv( v, g, b ):
    return g * ( 1 - np.exp( -b * v ) )


def mim_mim_iv( v, gp, bp, gn, bn ):
    return np.piecewise( v, [ v >= 0, v < 0 ],
                         [ lambda v: mim_iv( v, gp, bp ), lambda v: mim_iv( v, gn, bn ) ] )


def euler_step( x, t, f, dt, args ):
    return x + f( t, x, *args ) * dt


def rk4_step( x, t, f, dt, args ):
    k1 = f( t, x, *args )
    k2 = f( t + dt / 2, x + dt * k1 / 2, *args )
    k3 = f( t + dt / 2, x + dt * k2 / 2, *args )
    k4 = f( t + dt, x + dt * k3, *args )
    
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solver( f, time, dt, iv, args=[ ], method="Euler", I=None, I_args=None ):
    x_sol = [ iv ]
    
    if method == "Euler":
        step = euler_step
    if method == "RK4":
        step = rk4_step
    
    current = [ 0.0 ]
    
    for t in time[ 1: ]:
        if I:
            current.append( I( t, x_sol[ -1 ], *I_args ) )
        
        x = step( x_sol[ -1 ], t, f, dt, args )
        
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        
        x_sol.append( x )
    x_sol = np.array( x_sol )
    
    return (x_sol, current) if I else x_sol


def __animate_memristor( v, i, t, fig, axes, filename, axes_scale='linear' ):
    ax11 = axes[ 0 ]
    ax12 = axes[ 1 ]
    ax2 = axes[ 2 ]
    
    x11data, y11data = [ ], [ ]
    x12data, y12data = [ ], [ ]
    x2data, y2data = [ ], [ ]
    
    if axes_scale == 'linear':
        line11, = ax11.plot( t, i, color="b", animated=True )
        line2, = ax2.plot( v, i, color="b", animated=True )
    elif axes_scale == 'log':
        line11, = ax11.semilogy( t, i, color="b", animated=True )
        line2, = ax2.semilogy( v, i, color="b", animated=True )
    # -- voltage always on a linear scale
    line12, = ax12.plot( t, v, color="r", animated=True )
    
    def update( frame ):
        x11data.append( t[ frame ] )
        y11data.append( i[ frame ] )
        line11.set_data( x11data, y11data )
        x12data.append( t[ frame ] )
        y12data.append( v[ frame ] )
        line12.set_data( x12data, y12data )
        x2data.append( v[ frame ] )
        y2data.append( i[ frame ] )
        line2.set_data( x2data, y2data )
        
        return line11, line12, line2
    
    # Set up formatting for the movie files
    Writer = animation.writers[ 'ffmpeg' ]
    writer = Writer( fps=15, metadata=dict( artist='Me' ), bitrate=1800 )
    
    ani = animation.FuncAnimation( fig, update, frames=np.arange( 0, len( t ), 10 ), blit=True )
    ani.save( f"./videos/{filename}.mp4", writer=writer )
    
    return (line11, line12, line2)


def arrows( v, i, ax ):
    arrows_every = len( v ) // 200 if len( v ) > 200 else 1
    x1 = np.ma.masked_array( v[ :-1:arrows_every ], (np.diff( v ) > 0)[ ::arrows_every ] )
    x2 = np.ma.masked_array( v[ :-1:arrows_every ], (np.diff( v ) < 0)[ ::arrows_every ] )
    l1, = ax.plot( x1, i[ :-1:arrows_every ], 'b<' )
    l2, = ax.plot( x2, i[ :-1:arrows_every ], 'r>' )
    
    return l1, l2


def __plot_memristor( v, i, t, axes, iv_arrows, axes_scale='linear' ):
    ax11 = axes[ 0 ]
    ax12 = axes[ 1 ]
    ax2 = axes[ 2 ]
    
    line11, = ax11.plot( t, i, color="b" )
    line2, = ax2.plot( v, i, color="b" )
    # voltage always on a linear scale
    line12, = ax12.plot( t, v, color="r" )
    
    if iv_arrows:
        import matplotlib
        matplotlib.rcParams[ 'lines.markersize' ] = 5
        
        line2a1, line2a2 = arrows( v, i, ax2 )
    
    return (line11, line12, line2) if not iv_arrows else (line11, line12, line2, line2a1, line2a2)


def plot_memristor( v, i, t, title=None, figsize=(10, 4), iv_arrows=True, animated=False, filename=None, scaled=False,
                    axes_scale='linear', remove_noise=False ):
    i_oom = ("", "")
    t_oom = ("", "")
    if scaled:
        i_oom = order_of_magnitude.symbol( np.max( i ) )
        t_oom = order_of_magnitude.symbol( np.max( t ) )
        i = i * 1 / i_oom[ 0 ]
        t = t * 1 / t_oom[ 0 ]
    
    fig, axes = plt.subplots( 1, 2, figsize=figsize )
    ax11 = axes[ 0 ]
    ax12 = ax11.twinx()
    ax2 = axes[ 1 ]
    
    # if noise_threshold > 0.0:
    #     i[ np.isclose( i, np.zeros_like( i ), rtol=noise_threshold ) ] = 0
    if remove_noise:
        if axes_scale == 'linear':
            from scipy.signal import savgol_filter
            i = savgol_filter( i, 21, 3 )
        else:
            from tsmoothie.smoother import ConvolutionSmoother
            smoother = ConvolutionSmoother( window_len=20, window_type='blackman' )
            smoother.smooth( i )
            i = smoother.smooth_data[ 0 ]
    
    if axes_scale == 'log':
        i = np.abs( i )
        
        ax11.set_yscale( 'log' )
        ax2.set_yscale( 'log' )
    elif axes_scale == 'symlog':
        ax11.set_yscale( 'symlog', linthresh=1e-10, linscale=0.01 )
        ax2.set_yscale( 'symlog', linthresh=1e-10, linscale=0.01 )
    
    # filter out noise
    # apply log-modulus transform
    # i = np.sign( i ) * np.log10( np.abs( i ) + 1 )
    # i = np.where( i > 0, np.log10( i+1 ), np.where( i < 0, -np.log10( -i ), 1e-6 ) )
    # i = i + np.abs( np.min( i ) )
    
    ax11.set_ylabel( f"Current ({i_oom[ 1 ]}A)", color="b" )
    ax11.tick_params( 'y', colors='b' )
    ax11.set_xlim( np.min( t ), np.max( t ) )
    # ax11.set_ylim( [ np.min( i ) - np.abs( 0.5 * np.min( i ) ),
    #                  np.max( i ) + np.abs( 0.5 * np.max( i ) ) ] )
    ax11.set_xlabel( f"Time ({t_oom[ 1 ]}s)" )
    ax12.set_ylabel( 'Voltage (V)', color='r' )
    ax12.tick_params( 'y', colors='r' )
    ax12.set_xlim( np.min( t ), np.max( t ) )
    # ax12.set_ylim( [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
    ax2.set_xlim( [ np.min( v ) - np.abs( 0.5 * np.min( v ) ), np.max( v ) + np.abs( 0.5 * np.max( v ) ) ] )
    # ax2.set_ylim( [ np.min( i ) - np.abs( 0.5 * np.min( i ) ),
    #                 np.max( i ) + np.abs( 0.5 * np.max( i ) ) ] )
    ax2.set_ylabel( f"Current ({i_oom[ 1 ]}A)" )
    ax2.set_xlabel( "Voltage (V)" )
    if title:
        fig.suptitle( r"Nb-doped SrTiO$_3$ memristor (" + title + ")" )
    else:
        fig.suptitle( f"Memristor Voltage and Current vs. Time" )
    # fig.tight_layout()
    fig.subplots_adjust( left=0.1,
                         bottom=0.1,
                         right=0.9,
                         top=0.9,
                         wspace=0.4,
                         hspace=0.4 )
    
    if animated:
        lines = __animate_memristor( v, i, t, fig, [ ax11, ax12, ax2 ], filename, axes_scale=axes_scale )
    else:
        lines = __plot_memristor( v, i, t, [ ax11, ax12, ax2 ], iv_arrows, axes_scale=axes_scale )

    return fig, lines, (ax11, ax12, ax2)


def add_arrow_to_line2D( axes, line, arrow_locs=[ 0.2, 0.4, 0.6, 0.8 ], arrowstyle='-|>', arrowsize=1, transform=None ):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes:
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if not isinstance( line, mlines.Line2D ):
        raise ValueError( "expected a matplotlib.lines.Line2D object" )
    x, y = line.get_xdata(), line.get_ydata()
    
    arrow_kw = {
            "arrowstyle"    : arrowstyle,
            "mutation_scale": 10 * arrowsize,
            }
    
    color = line.get_color()
    use_multicolor_lines = isinstance( color, np.ndarray )
    if use_multicolor_lines:
        raise NotImplementedError( "multicolor lines not supported" )
    else:
        arrow_kw[ 'color' ] = color
    
    linewidth = line.get_linewidth()
    if isinstance( linewidth, np.ndarray ):
        raise NotImplementedError( "multiwidth lines not supported" )
    else:
        arrow_kw[ 'linewidth' ] = linewidth
    
    if transform is None:
        transform = axes.transData
    
    arrows = [ ]
    for loc in arrow_locs:
        s = np.cumsum( np.sqrt( np.diff( x )**2 + np.diff( y )**2 ) )
        n = np.searchsorted( s, s[ -1 ] * loc )
        arrow_tail = (x[ n ], y[ n ])
        arrow_head = (np.mean( x[ n:n + 2 ] ), np.mean( y[ n:n + 2 ] ))
        p = mpatches.FancyArrowPatch(
                arrow_tail, arrow_head, transform=transform,
                **arrow_kw )
        axes.add_patch( p )
        arrows.append( p )
    return arrows


# TODO pulsed breaks around 0 with certain frequency/time combinations
class InputVoltage():
    def __init__( self, shape=None, vp=None, vn=None, frequency=None, period=None, t_max=None ):
        self.shape = shape
        self.vp = vp
        self.vn = vn if vn else vp
        self.frequency = 1 / period if period else frequency
        self.period = 1 / frequency if frequency else period
        self.t_max = t_max
    
    def __call__( self, t ):
        pass
    
    def print( self, start="\t" ):
        start_lv2 = start + "\t"
        print( f"{start_lv2}Shape {self.shape}" )
        print( f"{start_lv2}Magnitude +{self.vp} / -{self.vn} V" )
        print( f"{start_lv2}Frequency {self.frequency} Hz" )
        print( f"{start_lv2}Period {self.period} s" )


class Interpolated( InputVoltage ):
    def __init__( self, x, y, degree=1 ):
        super( Interpolated, self ).__init__( "custom" )
        
        self.model = interpolate.splrep( x, y, s=0, k=degree )
    
    def __call__( self, t ):
        return interpolate.splev( t, self.model, der=0 )


class Sine( InputVoltage ):
    def __init__( self, vp=1, vn=None, frequency=None, period=None, t_max=0 ):
        assert frequency or period
        
        super( Sine, self ).__init__( "sine", vp, vn, frequency, period, t_max )
    
    def __call__( self, t ):
        pos = self.vp * np.sin( 2 * self.frequency * np.multiply( np.pi, t ) )
        neg = self.vn * np.sin( 2 * self.frequency * np.multiply( np.pi, t ) )
        v = np.where( pos > 0, pos, neg )
        
        return v


class Triangle( InputVoltage ):
    def __init__( self, vp=1, vn=None, frequency=None, period=None, t_max=0 ):
        assert frequency or period
        assert t_max > 0
        
        super( Triangle, self ).__init__( "triangle", vp, vn, frequency, period, t_max )
    
    def __call__( self, t ):
        pos = self.vp * np.abs( scipy.signal.sawtooth( 2 * self.frequency * np.pi * t + np.pi / 2, 0.5 ) )
        neg = -1 * self.vn * np.abs( scipy.signal.sawtooth( 2 * self.frequency * np.pi * t + np.pi / 2, 0.5 ) )
        
        if isinstance( t, np.ndarray ) and len( t ) > 1:
            pos[ len( pos ) // 2: ] *= -1
        elif t > self.t_max / 2:
            pos *= -1
        
        v = np.where( pos > 0, pos, neg )
        
        return v


class WindowFunction():
    def __init__( self, type, p=1, j=1 ):
        window_functions = {
                "none"    : self.no_window,
                "joglekar": self.joglekar,
                "biolek"  : self.biolek,
                "anusudha": self.anusudha,
                }
        
        assert type in [ "none", "joglekar", "biolek", "anusudha" ]
        self.type = type
        self.func = window_functions[ type ]
        self.p = p
        self.j = j
    
    def no_window( self, **kwargs ):
        return 1
    
    def joglekar( self, **kwargs ):
        x = kwargs[ "x" ]
        
        return 1 - np.power( np.multiply( 2, x ) - 1, 2 * self.p )
    
    def biolek( self, **kwargs ):
        x = kwargs[ "x" ]
        i = kwargs[ "i" ]
        
        return 1 - np.power( x - np.heaviside( -i, 1 ), 2 * self.p )
    
    def anusudha( self, **kwargs ):
        x = kwargs[ "x" ]
        
        return np.multiply( self.j, 1 - np.multiply( 2, np.power( np.power( x, 3 ) - x + 1, self.p ) ) )
    
    def print( self, start="\t" ):
        start_lv2 = start + "\t"
        print( f"{start_lv2}Type {self.type}" )
        print( f"{start_lv2}Parameter p {self.p}" )
        if self.type in ("anusudha"):
            print( f"{start_lv2}Parameter j {self.j}" )
