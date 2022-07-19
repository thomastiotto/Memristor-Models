import numpy as np
from decimal import Decimal
import math

from nengo.neurons import LIF
from nengo.params import NumberParam
from nengo.dists import Uniform, Choice
from nengo.utils.numpy import clip


class AdaptiveLIFLateralInhibition( LIF ):
    state = {
            "voltage"        : Uniform( low=0, high=1 ),
            "refractory_time": Choice( [ 0 ] ),
            "adaptation"     : Choice( [ 0 ] ),
            "inhibition"     : Choice( [ 0 ] )
            }
    spiking = True
    
    tau_n = NumberParam( "tau_n", low=0, low_open=True )
    inc_n = NumberParam( "inc_n", low=0 )
    tau_inhibition = NumberParam( "tau_inhibition", low=0 )
    
    def __init__(
            self,
            tau_n=1,
            inc_n=0.01,
            tau_rc=0.02,
            tau_ref=0.002,
            min_voltage=0,
            amplitude=1,
            initial_state=None,
            tau_inhibition=10,
            reset_every=0.35
            ):
        super().__init__(
                tau_rc=tau_rc,
                tau_ref=tau_ref,
                min_voltage=min_voltage,
                amplitude=amplitude,
                initial_state=initial_state,
                )
        self.tau_n = tau_n
        self.inc_n = inc_n
        self.tau_inhibition = tau_inhibition
        self.reset_every = reset_every
        self.sim_time = 0.0
    
    def step( self, dt, J, output, voltage, refractory_time, adaptation, inhibition ):
        """Implement the AdaptiveLIF nonlinearity."""
        
        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage
        tau_inhibition = self.tau_inhibition
        tau_ref = self.tau_ref
        tau_n = self.tau_n
        inc_n = self.inc_n
        reset_every = self.reset_every
        
        # reduce input by the adaptation
        J = J - adaptation
        
        # reset neurons (except for adaptation) after each sample is presented
        self.sim_time += dt
        if math.isclose( math.fmod( self.sim_time, reset_every ), 0, abs_tol=1e-3 ):
            J[ ... ] = 0
            voltage[ ... ] = 0
            refractory_time[ ... ] = 0
            inhibition[ ... ] = 0
        
        # reduce all refractory times by dt
        refractory_time -= dt
        
        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip( (dt - refractory_time), 0, dt )
        
        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1( -delta_t / tau_rc )
        
        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[ : ] = spiked_mask * (self.amplitude / dt)
        
        # if neuron that spiked had highest input but was still inhibited from a previous timestep
        voltage[ inhibition != 0 ] = 0
        output[ inhibition != 0 ] = 0
        spiked_mask[ inhibition != 0 ] = False
        
        if np.count_nonzero( output ) > 0:
            # inhibit all other neurons than one with highest input
            voltage[ J != np.max( J ) ] = 0
            output[ J != np.max( J ) ] = 0
            spiked_mask[ J != np.max( J ) ] = False
            inhibition[ (J != np.max( J )) & (inhibition == 0) ] = tau_inhibition
        
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
                -(voltage[ spiked_mask ] - 1) / (J[ spiked_mask ] - 1)
                )
        
        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[ voltage < min_voltage ] = min_voltage
        voltage[ spiked_mask ] = 0
        refractory_time[ spiked_mask ] = tau_ref + t_spike
        
        adaptation += (dt / tau_n) * (inc_n * output - adaptation)
        
        inhibition[ inhibition != 0 ] -= 1


import tensorflow as tf
from nengo_dl.neuron_builders import SoftLIFRateBuilder, LIFRateBuilder, LIFBuilder


class AdaptiveLIFLateralInhibitionBuilder( SoftLIFRateBuilder ):
    """Build a group of `~nengo.LIF` neuron operators."""
    
    spiking = True
    
    def build_pre( self, signals, config ):
        # note: we skip the SoftLIFRateBuilder init
        # pylint: disable=bad-super-call
        super( SoftLIFRateBuilder, self ).build_pre( signals, config )
        
        self.min_voltage = signals.op_constant(
                [ op.neurons for op in self.ops ],
                [ op.J.shape[ 0 ] for op in self.ops ],
                "min_voltage",
                signals.dtype,
                )
        self.tau_n = signals.op_constant(
                [ op.neurons for op in self.ops ],
                [ op.J.shape[ 0 ] for op in self.ops ],
                "tau_n",
                signals.dtype,
                )
        self.inc_n = signals.op_constant(
                [ op.neurons for op in self.ops ],
                [ op.J.shape[ 0 ] for op in self.ops ],
                "inc_n",
                signals.dtype,
                )
        self.tau_inhibition = signals.op_constant(
                [ op.neurons for op in self.ops ],
                [ op.J.shape[ 0 ] for op in self.ops ],
                "tau_inhibition",
                signals.dtype,
                )
    
    def step( self, J, dt, voltage, refractory_time, adaptation, inhibition ):
        """Implement the AdaptiveLIF nonlinearity."""
        
        def inhibit( voltage, output, inhibition, spiked_mask ):
            # inhibit all other neurons than one with highest input
            J_mask = tf.equal( J, tf.reduce_max( J ) )
            
            voltage = tf.multiply( voltage, tf.cast( J_mask, voltage.dtype ) )
            output = tf.multiply( output, tf.cast( J_mask, output.dtype ) )
            spiked_mask = tf.logical_and( spiked_mask, tf.cast( J_mask, spiked_mask.dtype ) )
            inhibition = tf.where( tf.logical_and( tf.logical_not( J_mask ), tf.equal( inhibition, 0 ) ),
                                   self.tau_inhibition,
                                   inhibition
                                   )
            
            return voltage, output, inhibition, spiked_mask
        
        J = J - adaptation
        
        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = tf.clip_by_value( dt - refractory_time, self.zero, dt )
        
        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        dV = (voltage - J) * tf.math.expm1(
                -delta_t / self.tau_rc  # pylint: disable=invalid-unary-operand-type
                )
        voltage += dV
        
        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > self.one
        output = tf.cast( spiked_mask, J.dtype ) * self.alpha
        
        inhibition_mask = tf.equal( inhibition, 0 )
        # if neuron that spiked had highest input but was still inhibited from a previous timestep
        voltage = tf.multiply( voltage, tf.cast( inhibition_mask, voltage.dtype ) )
        output = tf.multiply( output, tf.cast( inhibition_mask, output.dtype ) )
        spiked_mask = tf.logical_and( spiked_mask, tf.cast( inhibition_mask, spiked_mask.dtype ) )
        
        # inhibit all other neurons than one with highest input
        voltage, output, inhibition, spiked_mask = tf.cond( tf.math.count_nonzero( output ) > 0,
                                                            lambda: inhibit( voltage, output, inhibition, spiked_mask ),
                                                            lambda: (tf.identity( voltage ),
                                                                     tf.identity( output ),
                                                                     tf.identity( inhibition ),
                                                                     tf.identity( spiked_mask )
                                                                     )
                                                            )
        
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + self.tau_rc * tf.math.log1p( -(voltage - 1) / (J - 1) )
        
        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage = tf.where( spiked_mask, self.zeros, tf.maximum( voltage, self.min_voltage ) )
        refractory_time = tf.where( spiked_mask, self.tau_ref + t_spike, refractory_time - dt )
        
        adaptation += (dt / self.tau_n) * (self.inc_n * output - adaptation)
        
        inhibition_mask = tf.not_equal( inhibition, 0 )
        inhibition = tf.tensor_scatter_nd_sub( inhibition,
                                               tf.where( inhibition_mask ),
                                               tf.ones( tf.math.count_nonzero( inhibition_mask ) )
                                               )
        
        return output, voltage, refractory_time, adaptation, inhibition
    
    def training_step( self, J, dt, **state ):
        return (
                LIFRateBuilder.step( self, J, dt )
                if self.config.lif_smoothing is None
                else SoftLIFRateBuilder.step( self, J, dt )
        )


# register with the NengoDL neuron builder
from nengo_dl.neuron_builders import SimNeuronsBuilder

SimNeuronsBuilder.TF_NEURON_IMPL[ AdaptiveLIFLateralInhibition ] = AdaptiveLIFLateralInhibitionBuilder
