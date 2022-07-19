import datetime
import os

import matplotlib.pyplot as plt
import nengo
import nengo_dl
import numpy as np
from nengo.processes import Process
from nengo.params import NdarrayParam, NumberParam
from nengo.utils.matplotlib import rasterplot
import tensorflow as tf


def modify_learning_rate( sim, conn, rule, new_lr ):
    weights_sig = sim.model.sig[ conn ][ "weights" ]
    ops = [
            op
            for op in sim.model.operators
            if isinstance( op, rule ) and op.weights == weights_sig
            ]
    assert len( ops ) == 1
    op = ops[ 0 ]
    op.learning_rate = new_lr


def combine_probes( simulators ):
    # fail if we have multiple models
    assert all( [ sim.model.label == simulators[ 0 ].model.label for sim in simulators ] )
    
    model = simulators[ 0 ].model
    
    probes_comb = { }
    for probe in model.probes:
        concat = None
        for sim in simulators:
            if concat is None:
                concat = sim.data[ probe ]
            else:
                concat = np.concatenate( (concat, sim.data[ probe ]) )
        probes_comb[ probe ] = concat
    
    return probes_comb


def combine_tranges( simulators ):
    # fail if we have multiple timescales
    assert all( [ sim.dt == simulators[ 0 ].dt for sim in simulators ] )
    
    dt = simulators[ 0 ].dt
    
    concat = np.array( [ ] )
    for sim in simulators:
        concat = np.concatenate( (concat, sim.trange() + len( concat ) * dt) )
    
    return concat


def neural_activity_plot( probe, trange ):
    fig, ax = plt.subplots( figsize=(12.8, 7.2), dpi=100 )
    rasterplot( trange, probe, ax )
    ax.set_ylabel( 'Neuron' )
    ax.set_xlabel( 'Time (s)' )
    fig.get_axes()[ 0 ].annotate( "Pre" + " neural activity", (0.5, 0.94),
                                  xycoords='figure fraction', ha='center',
                                  fontsize=20
                                  )
    return fig


def heatmap_onestep( probe, t=-1, title="Weights after learning" ):
    if probe.shape[ 1 ] > 100:
        print( "Too many neurons to generate heatmap" )
        return

    cols = 10 if probe.shape[ 1 ] >= 10 else probe.shape[ 1 ]
    rows = (probe.shape[ 1 ] / cols) + 1 if probe.shape[ 1 ] % cols != 0 else probe.shape[ 1 ] / cols

    plt.set_cmap( 'jet' )
    fig, axes = plt.subplots( int( rows ), int( cols ), figsize=(12.8, 1.76 * rows), dpi=100 )
    for i, ax in enumerate( axes.flatten() if isinstance( axes, np.ndarray ) else [ axes ] ):
        try:
            ax.matshow( probe[ t, i, ... ].reshape( (28, 28) ) )
            ax.set_title( f"N. {i}" )
            ax.set_yticks( [ ] )
            ax.set_xticks( [ ] )
        except:
            ax.set_visible( False )
    fig.suptitle( title )
    fig.tight_layout()

    return fig


def generate_heatmap( probe, folder, sampled_every, num_samples=None ):
    from datetime import datetime
    
    if probe.shape[ 1 ] > 100:
        print( "Too many neurons to generate heatmap" )
        return
    
    folder_id = str( datetime.now().microsecond )
    try:
        os.makedirs( folder + folder_id )
    except FileExistsError:
        pass

    num_samples = num_samples if num_samples else probe.shape[ 0 ]
    step = int( probe.shape[ 0 ] / num_samples )

    print( "Saving Heatmaps ..." )
    for i in range( 0, probe.shape[ 0 ], step ):
        print( f"Saving {i} of {num_samples} images", end='\r' )
        fig = heatmap_onestep( probe, t=i, title=f"t={np.rint( i * sampled_every )}" )
        fig.savefig( folder + folder_id + "/" + str( i ).zfill( 10 ) + ".png", transparent=True, dpi=100 )
        plt.close()

    # to ensure we find ffmpeg
    os.environ[ "PATH" ] += os.pathsep + os.pathsep.join( [ "/opt/homebrew/bin" ] )

    print( "Generating Video from Heatmaps ..." )
    os.system(
            "ffmpeg "
            "-pattern_type glob -i '" + folder + folder_id + "/" + "*.png' "
                                                                   "-c:v libx264 -preset veryslow -crf 17 "
                                                                   "-tune stillimage -hide_banner -loglevel warning "
                                                                   "-y -pix_fmt yuv420p "
            + folder + "weight_evolution" + folder_id + ".mp4" )
    print( "Saved video in", folder )

    if os.path.isfile( folder + "weight_evolution" + folder_id + ".mp4" ):
        os.system( "rm -R " + folder + folder_id )


def pprint_dict( d, level=0 ):
    for k, v in d.items():
        if isinstance( v, dict ):
            print( "\t" * level, f"{k}:" )
            pprint_dict( v, level=level + 1 )
        else:
            print( "\t" * level, f"{k}: {v}" )


def setup():
    import sys
    
    os.environ[ "CUDA_DEVICE_ORDER" ] = "PCI_BUS_ID"
    os.environ[ 'TF_FORCE_GPU_ALLOW_GROWTH' ] = "true"
    
    # for nengo GUI
    sys.path.append( "." )
    # for rosa
    sys.path.append( ".." )
    
    tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )


class PresentInputWithPause( Process ):
    """Present a series of inputs, each for the same fixed length of time.

    Parameters
    ----------
    inputs : array_like
        Inputs to present, where each row is an input. Rows will be flattened.
    presentation_time : float
        Show each input for this amount of time (in seconds).
    """
    
    inputs = NdarrayParam( "inputs", shape=("...",) )
    presentation_time = NumberParam( "presentation_time", low=0, low_open=True )
    pause_time = NumberParam( "pause_time", low=0, low_open=True )
    
    def __init__( self, inputs, presentation_time, pause_time, **kwargs ):
        self.inputs = inputs
        self.presentation_time = presentation_time
        self.pause_time = pause_time
        
        super().__init__(
                default_size_in=0, default_size_out=self.inputs[ 0 ].size, **kwargs
                )
    
    def make_step( self, shape_in, shape_out, dt, rng, state ):
        assert shape_in == (0,)
        assert shape_out == (self.inputs[ 0 ].size,)
        
        n = len( self.inputs )
        inputs = self.inputs.reshape( n, -1 )
        presentation_time = float( self.presentation_time )
        pause_time = float( self.pause_time )
        
        def step_presentinput( t ):
            total_time = presentation_time + pause_time
            i = int( (t - dt) / total_time + 1e-7 )
            ti = t % total_time
            return np.zeros_like( inputs[ 0 ] ) if ti > presentation_time else inputs[ i % n ]
        
        return step_presentinput


class Sines( Process ):
    
    def __init__( self, period=4, **kwargs ):
        super().__init__( default_size_in=0, **kwargs )
        
        self.period = period
    
    def make_step( self, shape_in, shape_out, dt, rng, state ):
        # iteratively build phase shifted sines
        s = "lambda t: ("
        phase_shift = (2 * np.pi) / shape_out[ 0 ]
        for i in range( shape_out[ 0 ] ):
            s += f"np.sin( 1 / {self.period} * 2 * np.pi * t + {i * phase_shift}),"
        s += ")"
        signal = eval( s )
        
        def step_sines( t ):
            return signal( t )
        
        return step_sines


class SwitchInputs( Process ):
    def __init__( self, pre_switch, post_switch, switch_time, **kwargs ):
        assert issubclass( pre_switch.__class__, Process ) and issubclass( post_switch.__class__, Process ), \
            f"Expected two nengo Processes, got ({pre_switch.__class__},{post_switch.__class__}) instead"
        
        super().__init__( default_size_in=0, **kwargs )
        
        self.switch_time = switch_time
        self.preswitch_signal = pre_switch
        self.postswitch_signal = post_switch
    
    def make_step( self, shape_in, shape_out, dt, rng, state ):
        preswitch_step = self.preswitch_signal.make_step( shape_in, shape_out, dt, rng, state )
        postswitch_step = self.postswitch_signal.make_step( shape_in, shape_out, dt, rng, state )
        
        def step_switchinputs( t ):
            return preswitch_step( t ) if t < self.switch_time else postswitch_step( t )
        
        return step_switchinputs


class ConditionalProbe:
    def __init__( self, obj, attr, probe_from ):
        if isinstance( obj, nengo.Ensemble ):
            self.size_out = obj.dimensions
        if isinstance( obj, nengo.Node ):
            self.size_out = obj.size_out
        if isinstance( obj, nengo.Connection ):
            self.size_out = obj.size_out
        
        self.attr = attr
        self.time = probe_from
        self.probed_data = [ [ ] for _ in range( self.size_out ) ]
    
    def __call__( self, t, x ):
        if x.shape != (self.size_out,):
            raise RuntimeError(
                    "Expected dimensions=%d; got shape: %s"
                    % (self.size_out, x.shape)
                    )
        if t > 0 and t > self.time:
            for i, k in enumerate( x ):
                self.probed_data[ i ].append( k )
    
    @classmethod
    def setup( cls, obj, attr=None, probe_from=0 ):
        cond_probe = ConditionalProbe( obj, attr, probe_from )
        output = nengo.Node( cond_probe, size_in=cond_probe.size_out )
        nengo.Connection( obj, output, synapse=0.01 )
        
        return cond_probe
    
    def get_conditional_probe( self ):
        return np.array( self.probed_data ).T


class Plotter():
    def __init__( self, trange, rows, cols, dimensions, learning_time, sampling, plot_size=(12, 8), dpi=80, dt=0.001,
                  pre_alpha=0.3 ):
        self.time_vector = trange
        self.plot_sizes = plot_size
        self.dpi = dpi
        self.n_rows = rows
        self.n_cols = cols
        self.n_dims = dimensions
        self.learning_time = learning_time
        self.sampling = sampling
        self.dt = dt
        self.pre_alpha = pre_alpha
    
    def plot_testing( self, pre, post, smooth=False ):
        fig, axes = plt.subplots( 1, 1, sharex=True, sharey=True, squeeze=False )
        fig.set_size_inches( self.plot_sizes )
        
        learning_time = int( (self.learning_time / self.dt) / (self.sampling / self.dt) )
        time = self.time_vector[ learning_time:, ... ]
        pre = pre[ learning_time:, ... ]
        post = post[ learning_time:, ... ]
        
        axes[ 0, 0 ].xaxis.set_tick_params( labelsize='xx-large' )
        axes[ 0, 0 ].yaxis.set_tick_params( labelsize='xx-large' )
        axes[ 0, 0 ].set_ylim( -1, 1 )
        
        if smooth:
            from scipy.signal import savgol_filter
            
            pre = np.apply_along_axis( savgol_filter, 0, pre, window_length=51, polyorder=3 )
            post = np.apply_along_axis( savgol_filter, 0, post, window_length=51, polyorder=3 )
        
        axes[ 0, 0 ].plot(
                time,
                pre,
                # linestyle=":",
                alpha=self.pre_alpha,
                label='Pre' )
        axes[ 0, 0 ].set_prop_cycle( None )
        axes[ 0, 0 ].plot(
                time,
                post,
                label='Post' )
        # if self.n_dims <= 3:
        #     axes[ 0, 0 ].legend(
        #             [ f"Pre dim {i}" for i in range( self.n_dims ) ] +
        #             [ f"Post dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        # axes[ 0, 0 ].set_title( "Pre and post decoded on testing phase", fontsize=16 )
        
        plt.tight_layout()
        
        return fig
    
    def plot_results( self, input, pre, post, error, smooth=False ):
        fig, axes = plt.subplots( 3, 1, sharex=True, sharey=True, squeeze=False )
        fig.set_size_inches( self.plot_sizes )
        
        for ax in axes.flatten():
            ax.xaxis.set_tick_params( labelsize='xx-large' )
            ax.yaxis.set_tick_params( labelsize='xx-large' )
        
        axes[ 0, 0 ].plot(
                self.time_vector,
                input,
                label='Input',
                linewidth=2.0 )
        # if self.n_dims <= 3:
        #     axes[ 0, 0 ].legend(
        #             [ f"Input dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        axes[ 0, 0 ].set_title( "Input signal", fontsize=16 )
        
        if smooth:
            from scipy.signal import savgol_filter
            
            pre = np.apply_along_axis( savgol_filter, 0, pre, window_length=51, polyorder=3 )
            post = np.apply_along_axis( savgol_filter, 0, post, window_length=51, polyorder=3 )
        
        axes[ 1, 0 ].plot(
                self.time_vector,
                pre,
                # linestyle=":",
                alpha=self.pre_alpha,
                label='Pre' )
        axes[ 1, 0 ].set_prop_cycle( None )
        axes[ 1, 0 ].plot(
                self.time_vector,
                post,
                label='Post' )
        # if self.n_dims <= 3:
        #     axes[ 1, 0 ].legend(
        #             [ f"Pre dim {i}" for i in range( self.n_dims ) ] +
        #             [ f"Post dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        axes[ 1, 0 ].set_title( "Pre and post decoded", fontsize=16 )
        
        if smooth:
            from scipy.signal import savgol_filter
            
            error = np.apply_along_axis( savgol_filter, 0, error, window_length=51, polyorder=3 )
        axes[ 2, 0 ].plot(
                self.time_vector,
                error,
                label='Error' )
        if self.n_dims <= 3:
            axes[ 2, 0 ].legend(
                    [ f"Error dim {i}" for i in range( self.n_dims ) ],
                    loc='best' )
        axes[ 2, 0 ].set_title( "Error", fontsize=16 )
        
        for ax in axes:
            ax[ 0 ].axvline( x=self.learning_time, c="k" )
        
        fig.get_axes()[ 0 ].annotate( f"{self.n_rows} neurons, {self.n_dims} dimensions", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=20
                                      )
        plt.tight_layout()
        
        return fig
    
    def plot_ensemble_spikes( self, name, spikes, decoded ):
        fig, ax1 = plt.subplots()
        fig.set_size_inches( self.plot_sizes )
        ax1 = plt.subplot( 1, 1, 1 )
        rasterplot( self.time_vector, spikes, ax1 )
        ax1.axvline( x=self.learning_time, c="k" )
        ax2 = plt.twinx()
        ax2.plot( self.time_vector, decoded, c="k", alpha=0.3 )
        ax1.set_xlim( 0, max( self.time_vector ) )
        ax1.set_ylabel( 'Neuron' )
        ax1.set_xlabel( 'Time (s)' )
        fig.get_axes()[ 0 ].annotate( name + " neural activity", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=20
                                      )

        return fig

    def plot_values_over_time( self, pos_memr, neg_memr, value="conductance", plot_all=False ):
        plot_range = int( self.learning_time / self.dt ) if not plot_all else int( len( self.time_vector ) )
    
        if value == "conductance":
            tit = "Conductances"
            pos_memr = 1 / pos_memr
            neg_memr = 1 / neg_memr
        if value == "resistance":
            tit = "Resistances"
        fig, axes = plt.subplots( self.n_rows, self.n_cols )
        fig.set_size_inches( self.plot_sizes )
        for i in range( axes.shape[ 0 ] ):
            for j in range( axes.shape[ 1 ] ):
                pos = pos_memr[ :plot_range, i, j ]
                neg = neg_memr[ :plot_range, i, j ]
                axes[ i, j ].plot( pos, c="r" )
                axes[ i, j ].plot( neg, c="b" )
                axes[ i, j ].set_title( f"{j}->{i}" )
                axes[ i, j ].set_yticklabels( [ ] )
                # axes[ i, j ].set_xticklabels( [ ] )
                plt.subplots_adjust( hspace=0.7 )
        fig.get_axes()[ 0 ].annotate( f"{tit} over time", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=20
                                      )
        # plt.tight_layout()
    
        return fig

    def plot_weights_over_time( self, pos_memr, neg_memr, plot_all=False ):
        plot_range = int( self.learning_time / self.dt ) if not plot_all else int( len( self.time_vector ) )
    
        fig, axes = plt.subplots( self.n_rows, self.n_cols )
        fig.set_size_inches( self.plot_sizes )
        for i in range( axes.shape[ 0 ] ):
            for j in range( axes.shape[ 1 ] ):
                pos = 1 / pos_memr[ :plot_range, i, j ]
                neg = 1 / neg_memr[ :plot_range, i, j ]
                axes[ i, j ].plot( pos - neg, c="g" )
                axes[ i, j ].set_title( f"{j}->{i}" )
                axes[ i, j ].set_yticklabels( [ ] )
                # axes[ i, j ].set_xticklabels( [ ] )
                plt.subplots_adjust( hspace=0.7 )
        fig.get_axes()[ 0 ].annotate( "Weights over time", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=20
                                      )
        # plt.tight_layout()
        
        return fig
    
    def plot_weight_matrices_over_time( self, weights, n_cols=5, sample_every=0.001 ):
        n_rows = int( self.learning_time / n_cols ) + 1
        fig, axes = plt.subplots( n_rows, n_cols )
        fig.set_size_inches( self.plot_sizes )
        
        for t, ax in enumerate( axes.flatten() ):
            if t <= self.learning_time:
                ax.matshow( weights[ int( (t / self.dt) / (sample_every / self.dt) ), ... ],
                            cmap=plt.cm.Blues )
                ax.set_title( f"{t}" )
                ax.set_yticklabels( [ ] )
                ax.set_xticklabels( [ ] )
                plt.subplots_adjust( hspace=0.7 )
            else:
                ax.set_axis_off()
        fig.get_axes()[ 0 ].annotate( "Weights over time", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=18
                                      )
        # plt.tight_layout()
        
        return fig


def make_timestamped_dir( root=None ):
    if root is None:
        root = "../data/"

    os.makedirs( os.path.dirname( root ), exist_ok=True )

    time_string = datetime.datetime.now().strftime( "%d-%m-%Y_%H-%M-%S" )
    dir_name = root + "/" + time_string + "/"
    if os.path.isdir( dir_name ):
        raise FileExistsError( "The directory already exists" )
    dir_images = dir_name + "images/"
    dir_data = dir_name + "data/"
    os.makedirs( dir_name )
    os.makedirs( dir_images )
    os.makedirs( dir_data )

    return dir_name, dir_images, dir_data


def mse_to_rho_ratio( mse, rho ):
    return [ i for i in np.array( rho ) / mse ]


def correlations( X, Y ):
    import scipy
    
    pearson_correlations = [ ]
    spearman_correlations = [ ]
    kendall_correlations = [ ]
    for x, y in zip( X.T, Y.T ):
        pearson_correlations.append( scipy.stats.pearsonr( x, y )[ 0 ] )
        spearman_correlations.append( scipy.stats.spearmanr( x, y )[ 0 ] )
        kendall_correlations.append( scipy.stats.kendalltau( x, y )[ 0 ] )
    
    return pearson_correlations, spearman_correlations, kendall_correlations


def gini( array ):
    """Calculate the Gini coefficient of exponent numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin( array ) < 0:
        # Values cannot be negative:
        array -= np.amin( array )
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort( array )
    # Index per array element:
    index = np.arange( 1, array.shape[ 0 ] + 1 )
    # Number of array elements:
    n = array.shape[ 0 ]
    # Gini coefficient:
    return ((np.sum( (2 * index - n - 1) * array )) / (n * np.sum( array )))


def save_weights( path, probe ):
    np.save( path + "weights.npy", probe[ -1 ].T )


def save_memristors_to_csv( dir, pos_memr, neg_memr ):
    num_post = pos_memr.shape[ 0 ]
    num_pre = pos_memr.shape[ 1 ]
    
    pos_memr = pos_memr.reshape( (pos_memr.shape[ 0 ], -1) )
    neg_memr = neg_memr.reshape( (neg_memr.shape[ 0 ], -1) )
    
    header = [ ]
    for i in range( num_post ):
        for j in range( num_pre ):
            header.append( f"{j}->{i}" )
    header = ','.join( header )
    
    np.savetxt( dir + "pos_resistances.csv", pos_memr, delimiter=",", header=header, comments="" )
    np.savetxt( dir + "neg_resistances.csv", neg_memr, delimiter=",", header=header, comments="" )
    np.savetxt( dir + "weights.csv", 1 / pos_memr - 1 / neg_memr, delimiter=",", header=header, comments="" )


def save_results_to_csv( dir, input, pre, post, error ):
    header = [ ]
    header.append( ",".join( [ "input" + str( i ) for i in range( input.shape[ 1 ] ) ] ) )
    header.append( ",".join( [ "pre" + str( i ) for i in range( pre.shape[ 1 ] ) ] ) )
    header.append( ",".join( [ "post" + str( i ) for i in range( post.shape[ 1 ] ) ] ) )
    header.append( ",".join( [ "error" + str( i ) for i in range( error.shape[ 1 ] ) ] ) )
    header = ",".join( header )
    
    with open( dir + "results.csv", "w" ) as f:
        np.savetxt( f, np.hstack( (input, pre, post, error) ), delimiter=",", header=header, comments="" )


def nested_dict( n, type ):
    from collections import defaultdict
    
    if n == 1:
        return defaultdict( type )
    else:
        return defaultdict( lambda: nested_dict( n - 1, type ) )
