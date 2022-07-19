import warnings
import numpy as np

from nengo.builder import Operator
from nengo.builder.learning_rules import build_or_passthrough, get_post_ens, get_pre_ens
from nengo.learning_rules import LearningRuleType
from nengo.params import Default, NumberParam, DictParam
from nengo.synapses import Lowpass, SynapseParam

from memristor_nengo.debug_plots import debugger_plots
from memristor_nengo.yakopcic_functions import *


def initialise_memristors2(rule, in_size, out_size):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(rule.seed)
        r_min_noisy = get_truncated_normal(rule.r_min, rule.r_min * rule.noise_percentage[0],
                                           0, np.inf, out_size, in_size)
        np.random.seed(rule.seed)
        r_max_noisy = get_truncated_normal(rule.r_max, rule.r_max * rule.noise_percentage[1],
                                           np.max(r_min_noisy), np.inf, out_size, in_size)
        exponent_noisy = 0

    np.random.seed(rule.seed)
    pos_mem_initial = np.full((out_size, in_size), 1e8)
    neg_mem_initial = np.full((out_size, in_size), 1e8)

    pos_memristors = Signal(shape=(out_size, in_size), name=f"{rule}:pos_memristors",
                            initial_value=pos_mem_initial)
    neg_memristors = Signal(shape=(out_size, in_size), name=f"{rule}:neg_memristors",
                            initial_value=neg_mem_initial)

    return pos_memristors, neg_memristors, r_min_noisy, r_max_noisy, exponent_noisy


def find_spikes(input_activities, shape, output_activities=None, invert=False):
    output_size = shape[0]
    input_size = shape[1]

    spiked_pre = np.tile(
        np.array(np.rint(input_activities), dtype=bool), (output_size, 1)
    )
    spiked_post = np.tile(
        np.expand_dims(
            np.array(np.rint(output_activities), dtype=bool), axis=1), (1, input_size)
    ) \
        if output_activities is not None \
        else np.ones((output_size, input_size))

    out = np.logical_and(spiked_pre, spiked_post)
    return out if not invert else np.logical_not(out)



def update_memristors2(update_steps, pos_memristors, neg_memristors, r):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos_memristors[update_steps > 0] -= r[update_steps > 0]
        neg_memristors[update_steps < 0] -= r[update_steps < 0]


def update_weights(V, weights, pos_memristors, neg_memristors, r_max, r_min, gain):
    weights[V > 0] = gain * \
                     (resistance2conductance(pos_memristors[V > 0], r_min[V > 0],
                                             r_max[V > 0])
                      - resistance2conductance(neg_memristors[V > 0], r_min[V > 0],
                                               r_max[V > 0]))
    weights[V < 0] = gain * \
                     (resistance2conductance(pos_memristors[V < 0], r_min[V < 0],
                                             r_max[V < 0])
                      - resistance2conductance(neg_memristors[V < 0], r_min[V < 0],
                                               r_max[V < 0]))


class mPES(LearningRuleType):
    modifies = "weights"
    probeable = ("error", "activities", "delta", "pos_memristors", "neg_memristors")

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    r_max = NumberParam("r_max", readonly=True, default=2.3e8)
    r_min = NumberParam("r_min", readonly=True, default=200)
    exponent = NumberParam("exponent", readonly=True, default=-0.146)
    gain = NumberParam("gain", readonly=True, default=1e3)
    voltage = NumberParam("voltage", readonly=True, default=1e-1)
    initial_state = DictParam("initial_state", optional=True)

    def __init__(self,
                 pre_synapse=Default,
                 r_max=Default,
                 r_min=Default,
                 exponent=Default,
                 noisy=False,
                 gain=Default,
                 voltage=Default,
                 initial_state=None,
                 seed=None):
        super().__init__(size_in="post_state")

        self.pre_synapse = pre_synapse
        self.r_max = r_max
        self.r_min = r_min
        self.exponent = exponent
        if not noisy:
            self.noise_percentage = np.zeros(4)
        elif isinstance(noisy, float) or isinstance(noisy, int):
            self.noise_percentage = np.full(4, noisy)
        elif isinstance(noisy, list) and len(noisy) == 4:
            self.noise_percentage = noisy
        else:
            raise ValueError(f"Noisy parameter must be int or list of length 4, not {type(noisy)}")
        self.gain = gain
        self.voltage = voltage
        self.seed = seed
        self.initial_state = {} if initial_state is None else initial_state

    @property
    def _argdefaults(self):
        return (
            ("pre_synapse", mPES.pre_synapse.default),
            ("r_max", mPES.r_max.default),
            ("r_min", mPES.r_min.default),
            ("exponent", mPES.exponent.default),
        )


class SimmPES(Operator):
    def __init__(
            self,
            pre_filtered,
            error,
            pos_memristors,
            neg_memristors,
            weights,
            noise_percentage,
            gain,
            r_min,
            r_max,
            exponent,
            bmax_n,
            bmax_p,
            bmin_n,
            bmin_p,
            gmax_n,
            gmax_p,
            gmin_n,
            gmin_p,
            Vn,
            Vp,
            An,
            Ap,
            noise,
            initial_state,
            tag=None
    ):
        super(SimmPES, self).__init__(tag=tag)

        self.noise_percentage = noise_percentage
        self.gain = gain
        self.error_threshold = 1e-5
        self.r_min = r_min
        self.r_max = r_max
        self.exponent = exponent
        self.An = An #0.1065893286 * 8e-1
        self.Ap = Ap #0.0118432587 * 4e1
        self.Vn = .5 #Vn
        self.Vp = .5 #Vp
        self.alphan = 1.0
        self.alphap = 1.0
        self.bmax_n = 3.23 #bmax_n
        self.bmax_p = 4.96 #bmax_p
        self.bmin_n = 2.6 #bmin_n
        self.bmin_p = 6.91 #bmin_p
        self.gmax_n = 0.00017 #gmax_n
        self.gmax_p = 9.0e-05 #gmax_p
        self.gmin_n = 4.4e-07 #gmin_n
        self.gmin_p = 1.5e-05 #gmin_p
        self.xn = 0.242
        self.xp = .1
        self.x = 0
        self.x_p = 0
        self.x_n = 0
        self.pulse = 0
        self.currents = []
        self.xs = []
        self.rs = []
        self.debug = False
        self.noise_percentage = noise
        self.initial_state = initial_state

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error]
        self.updates = [weights, pos_memristors, neg_memristors]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def error(self):
        return self.reads[1]

    @property
    def weights(self):
        return self.updates[0]

    @property
    def pos_memristors(self):
        return self.updates[1]

    @property
    def neg_memristors(self):
        return self.updates[2]

    def _descstr(self):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.weights)

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        local_error = signals[self.error]

        pos_memristors = signals[self.pos_memristors]
        neg_memristors = signals[self.neg_memristors]
        weights = signals[self.weights]

        gain = self.gain
        error_threshold = self.error_threshold
        r_min = self.r_min
        r_max = self.r_max
        exponent = self.exponent
        # print("shape of memristor: ", pos_memristors.shape)

        # overwrite initial transform with memristor-based weights
        if "weights" in self.initial_state:
            weights[:] = self.initial_state["weights"]
        else:
            weights[:] = gain * \
                         (resistance2conductance(pos_memristors, r_min, r_max)
                          - resistance2conductance(neg_memristors, r_min, r_max))

        self.min_error = self.max_error = 0
        pulse_levels = 100

        def step_simmpes():
            self.An = np.random.normal(self.An, self.An * self.noise_percentage, self.An.shape)
            self.Ap = np.random.normal(self.Ap, self.Ap * self.noise_percentage, self.Ap.shape)
            # set update to zero if error is small or adjustments go on for ever
            # if error is small return zero delta
            if np.any(np.absolute(local_error) > error_threshold):
                # calculate the magnitude of the update based on PES learning rule
                pes_delta = np.outer(-local_error, pre_filtered)

                # some memristors are adjusted erroneously if we don't filter
                spiked_map = find_spikes(pre_filtered, weights.shape, invert=True)
                pes_delta[spiked_map] = 0

                V = np.sign(pes_delta) * 6e-1
                #self.Vn = np.random.normal(self.Vn, self.Vn * self.noise_percentage, self.Vn.shape)
                #self.Vp = np.random.normal(self.Vp, self.Vp * self.noise_percentage, self.Vp.shape)
                # print("V: ", V, "\n")

                # Calculate the state variables at a current timestep
                np.seterr(all="raise")
                self.x = self.x + dxdt(np.abs(V), self.x, self.Ap, self.An, self.Vp, self.Vn,
                                       self.xp, self.xn, self.alphap, self.alphan, 1) * dt
                # Clip the value of state variables beyond the [0,1] range
                self.x = np.select([self.x < 0, self.x > 1], [0, 1], default=self.x)

                # Calculate the current and the resistance for the devices
                i = current(np.abs(V), self.x, self.gmax_p, self.bmax_p,
                            self.gmax_n, self.bmax_n, self.gmin_p, self.bmin_p, self.gmin_n, self.bmin_n)
                r = np.divide(np.abs(V), i, out=np.zeros(V.shape, dtype=float), where=i != 0)
                # Clip the value of resistances beyond the [r_min, r_max] range
                r = np.select([r < self.r_min, r > self.r_max], [r_min, r_max], default=r)

                # update the two memristor pairs
                update_memristors2(V, pos_memristors, neg_memristors, r)

                # update network weights
                update_weights(V, weights, pos_memristors, neg_memristors, r_max, r_min, gain)

                # Debugging plots, show currents and state variables over simtime.
                if self.pulse < 22000 and self.debug:
                    self.currents.append(i)
                    self.xs.append(self.x)
                    self.rs.append(r)
                    self.pulse += 1
                    print("Pulse no: ", self.pulse)

                elif self.pulse == 22000 and self.debug:
                    debugger_plots(self.currents, self.xs, self.rs, self.pulse)

                else:
                    self.pulse += 1
                    if self.debug:
                        print(self.pulse)

        return step_simmpes


"""
BUILDERS
These functions link the front-end to the back-end by initialising the Signals
"""

import tensorflow as tf
from nengo.builder import Signal
from nengo.builder.operator import Reset, DotInc, Copy

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder
from nengo.builder import Builder as NengoCoreBuilder


@NengoBuilder.register(mPES)
@NengoCoreBuilder.register(mPES)
def build_mpes(model, mpes, rule):
    conn = rule.connection

    # Create input error signal
    error = Signal(shape=(rule.size_in,), name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]["in"] = error  # error connection will attach here

    acts = build_or_passthrough(model, mpes.pre_synapse, model.sig[conn.pre_obj]["out"])

    post = get_post_ens(conn)
    encoders = model.sig[post]["encoders"]

    pos_memristors, neg_memristors, r_min_noisy, r_max_noisy, exponent_noisy = initialise_memristors2(mpes,
                                                                                                     acts.shape[0],
                                                                                                     encoders.shape[
                                                                                                         0])

    model.sig[conn]["pos_memristors"] = pos_memristors
    model.sig[conn]["neg_memristors"] = neg_memristors

    if conn.post_obj is not conn.post:
        # in order to avoid slicing encoders along an axis > 0, we pad
        # `error` out to the full base dimensionality and then do the
        # dotinc with the full encoder matrix
        # comes into effect when slicing post connection
        padded_error = Signal(shape=(encoders.shape[1],))
        model.add_op(Copy(error, padded_error, dst_slice=conn.post_slice))
    else:
        padded_error = error

    # error = dot(encoders, error)
    local_error = Signal(shape=(post.n_neurons,))
    model.add_op(Reset(local_error))
    model.add_op(DotInc(encoders, padded_error, local_error, tag="PES:encode"))

    bmax_n = np.random.normal(3.23, 3.23 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    bmax_p = np.random.normal(4.96, 4.96 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    bmin_n = np.random.normal(2.6, 2.6 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    bmin_p = np.random.normal(6.91, 6.91 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    gmax_n = np.random.normal(0.00017, 0.00017 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    gmax_p = np.random.normal(9.0e-05, 9.0e-05 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    gmin_n = np.random.normal(4.4e-07, 4.4e-07 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    gmin_p = np.random.normal(1.5e-05, 1.5e-05 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    Vn = np.random.normal(.5, .5 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    Vp = np.random.normal(.5, .5 * mpes.noise_percentage[3], (encoders.shape[0], acts.shape[0]))
    An = np.random.normal(0.1065893286 * 8e-1, 0.1065893286 * 8e-1 * mpes.noise_percentage[3],
                          (encoders.shape[0], acts.shape[0]))
    Ap = np.random.normal(0.0118432587 * 4e1, 0.0118432587 * 4e1 * mpes.noise_percentage[3],
                          (encoders.shape[0], acts.shape[0]))
    noise = mpes.noise_percentage[3]

    model.operators.append(
        SimmPES(acts,
                local_error,
                model.sig[conn]["pos_memristors"],
                model.sig[conn]["neg_memristors"],
                model.sig[conn]["weights"],
                mpes.noise_percentage,
                mpes.gain,
                r_min_noisy,
                r_max_noisy,
                exponent_noisy,
                bmax_n,
                bmax_p,
                bmin_n,
                bmin_p,
                gmax_n,
                gmax_p,
                gmin_n,
                gmin_p,
                Vn,
                Vp,
                An,
                Ap,
                noise,
                mpes.initial_state,
                )
    )

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts
    model.sig[rule]["pos_memristors"] = pos_memristors
    model.sig[rule]["neg_memristors"] = neg_memristors
