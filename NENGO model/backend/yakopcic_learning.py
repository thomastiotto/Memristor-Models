import warnings
from copy import copy

import numpy as np

from nengo.builder import Operator
from nengo.builder.learning_rules import build_or_passthrough, get_post_ens, get_pre_ens
from nengo.learning_rules import LearningRuleType
from nengo.params import Default, NumberParam, DictParam
from nengo.synapses import Lowpass, SynapseParam

from debug_plots import debugger_plots
from yakopcic_functions import *


def initialise_yakopcic_memristors(x0, bmax_n, bmax_p, bmin_n, bmin_p, gmax_n, gmax_p, gmin_n, gmin_p,
                                   in_size, out_size, name):
    V_read = -1
    i = current(V_read, x0, gmax_p, bmax_p,
                gmax_n, bmax_n, gmin_p, bmin_p, gmin_n, bmin_n)
    r = np.divide(V_read, i)

    memristors = Signal(shape=(out_size, in_size), name=name, initial_value=r)

    return memristors


def initialise_yakopcic_model(noise_percentage, encoders, acts, seed):
    np.random.seed(seed)

    # -- parameters fund with pulse_experiment_1s_to_1ms.py
    yakopcic_model = json.load(open('../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an'))

    An = get_truncated_normal(yakopcic_model['An'], yakopcic_model['An'] * noise_percentage,
                              0, np.inf,
                              encoders.shape[0], acts.shape[0])
    Ap = get_truncated_normal(yakopcic_model['Ap'], yakopcic_model['Ap'] * noise_percentage,
                              0, np.inf,
                              encoders.shape[0], acts.shape[0])
    Vn = get_truncated_normal(yakopcic_model['Vn'], yakopcic_model['Vn'] * noise_percentage,
                              0, np.inf,
                              encoders.shape[0], acts.shape[0])
    Vp = get_truncated_normal(yakopcic_model['Vp'], yakopcic_model['Vp'] * noise_percentage,
                              0, np.inf,
                              encoders.shape[0], acts.shape[0])
    alphan = get_truncated_normal(yakopcic_model['alphan'], yakopcic_model['alphan'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    alphap = get_truncated_normal(yakopcic_model['alphap'], yakopcic_model['alphap'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    bmax_n = get_truncated_normal(yakopcic_model['bmax_n'], yakopcic_model['bmax_n'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    bmax_p = get_truncated_normal(yakopcic_model['bmax_p'], yakopcic_model['bmax_p'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    bmin_n = get_truncated_normal(yakopcic_model['bmin_n'], yakopcic_model['bmin_n'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    bmin_p = get_truncated_normal(yakopcic_model['bmin_p'], yakopcic_model['bmin_p'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    gmax_n = get_truncated_normal(yakopcic_model['gmax_n'], yakopcic_model['gmax_n'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    gmax_p = get_truncated_normal(yakopcic_model['gmax_p'], yakopcic_model['gmax_p'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    gmin_n = get_truncated_normal(yakopcic_model['gmin_n'], yakopcic_model['gmin_n'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    gmin_p = get_truncated_normal(yakopcic_model['gmin_p'], yakopcic_model['gmin_p'] * noise_percentage,
                                  0, np.inf,
                                  encoders.shape[0], acts.shape[0])
    xn = get_truncated_normal(yakopcic_model['xn'], yakopcic_model['xn'] * noise_percentage,
                              0, np.inf,
                              encoders.shape[0], acts.shape[0])
    xp = get_truncated_normal(yakopcic_model['xp'], yakopcic_model['xp'] * noise_percentage,
                              0, np.inf,
                              encoders.shape[0], acts.shape[0])
    # -- obtained from pulse_experiment_1s_to_1ms.py
    x0 = get_truncated_normal(0.6251069761800688, 0.6251069761800688 * noise_percentage,
                              0, 1,
                              encoders.shape[0], acts.shape[0])

    return An, Ap, Vn, Vp, alphan, alphap, bmax_n, bmax_p, bmin_n, bmin_p, gmax_n, gmax_p, gmin_n, gmin_p, xn, xp, x0


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


class mPES(LearningRuleType):
    modifies = "weights"
    probeable = ("error", "activities", "delta", "pos_memristors", "neg_memristors")

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    gain = NumberParam("gain", readonly=True, default=1e3)
    voltage = NumberParam("voltage", readonly=True, default=0.1)
    initial_state = DictParam("initial_state", optional=True)

    def __init__(self,
                 pre_synapse=Default,
                 noisy=False,
                 gain=Default,
                 voltage=Default,
                 initial_state=None,
                 seed=None):
        super().__init__(size_in="post_state")

        self.pre_synapse = pre_synapse
        if not noisy:
            self.noise_percentage = 0
        elif isinstance(noisy, float) or isinstance(noisy, int):
            self.noise_percentage = noisy
        else:
            raise ValueError(f"Noisy parameter must be float or list of length 4, not {type(noisy)}")
        self.gain = gain
        self.voltage = voltage
        self.seed = seed
        self.initial_state = {} if initial_state is None else initial_state

    @property
    def _argdefaults(self):
        return (
            ("pre_synapse", mPES.pre_synapse.default)
        )


class SimmPES(Operator):
    def __init__(
            self,
            pre_filtered,
            error,
            pos_memristors,
            neg_memristors,
            weights,
            gain,
            bmax_n_pos,
            bmax_p_pos,
            bmin_n_pos,
            bmin_p_pos,
            gmax_n_pos,
            gmax_p_pos,
            gmin_n_pos,
            gmin_p_pos,
            Vn_pos,
            Vp_pos,
            alphan_pos,
            alphap_pos,
            An_pos,
            Ap_pos,
            x0_pos,
            xn_pos,
            xp_pos,
            bmax_n_neg,
            bmax_p_neg,
            bmin_n_neg,
            bmin_p_neg,
            gmax_n_neg,
            gmax_p_neg,
            gmin_n_neg,
            gmin_p_neg,
            Vn_neg,
            Vp_neg,
            alphan_neg,
            alphap_neg,
            An_neg,
            Ap_neg,
            x0_neg,
            xn_neg,
            xp_neg,
            initial_state,
            tag=None
    ):
        super(SimmPES, self).__init__(tag=tag)

        self.gain = gain
        self.error_threshold = 1e-5

        self.An_pos = An_pos
        self.Ap_pos = Ap_pos
        self.Vn_pos = Vn_pos
        self.Vp_pos = Vp_pos
        self.alphan_pos = alphan_pos
        self.alphap_pos = alphap_pos
        self.bmax_n_pos = bmax_n_pos
        self.bmax_p_pos = bmax_p_pos
        self.bmin_n_pos = bmin_n_pos
        self.bmin_p_pos = bmin_p_pos
        self.gmax_n_pos = gmax_n_pos
        self.gmax_p_pos = gmax_p_pos
        self.gmin_n_pos = gmin_n_pos
        self.gmin_p_pos = gmin_p_pos
        self.xn_pos = xn_pos
        self.xp_pos = xp_pos
        self.x_pos = x0_pos
        self.An_neg = An_neg
        self.Ap_neg = Ap_neg
        self.Vn_neg = Vn_neg
        self.Vp_neg = Vp_neg
        self.alphan_neg = alphan_neg
        self.alphap_neg = alphap_neg
        self.bmax_n_neg = bmax_n_neg
        self.bmax_p_neg = bmax_p_neg
        self.bmin_n_neg = bmin_n_neg
        self.bmin_p_neg = bmin_p_neg
        self.gmax_n_neg = gmax_n_neg
        self.gmax_p_neg = gmax_p_neg
        self.gmin_n_neg = gmin_n_neg
        self.gmin_p_neg = gmin_p_neg
        self.xn_neg = xn_neg
        self.xp_neg = xp_neg
        self.x_neg = x0_neg

        self.currents = []
        self.xs = []
        self.rs = []
        self.initial_state = initial_state

        self.pos_pulse_counter = np.zeros_like(x0_pos)
        self.neg_pulse_counter = np.zeros_like(x0_neg)
        self.pulse_archive = []

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

        # overwrite initial transform with memristor-based weights
        if "weights" in self.initial_state:
            weights[:] = self.initial_state["weights"]
        else:
            weights[:] = gain * (1 / pos_memristors - 1 / neg_memristors)

        def step_simmpes():
            # set update to zero if error is small or adjustments go on forever
            # if error is small return zero delta
            if np.any(np.absolute(local_error) > error_threshold):
                # calculate the magnitude of the update based on PES learning rule
                pes_delta = np.outer(-local_error, pre_filtered)

                # some memristors are adjusted erroneously if we don't filter
                spiked_map = find_spikes(pre_filtered, weights.shape, invert=True)
                pes_delta[spiked_map] = 0

                V = np.sign(pes_delta) * 3.86621037038006

                # -- count pulses given to each memristor
                pos_temp = np.sign(pes_delta)
                pos_temp[pos_temp < 0] = 0
                neg_temp = np.sign(pes_delta)
                neg_temp[neg_temp > 0] = 0
                neg_temp = np.abs(neg_temp)
                self.pos_pulse_counter += pos_temp
                self.neg_pulse_counter += neg_temp

                self.pulse_archive.append(np.sign(pes_delta))

                # print("V: ", V, "\n")
                def yakopcic_one_step(V, x, Ap, An, Vp, Vn, alphap, alphan, xp, xn):
                    # Calculate the state variables at the current timestep
                    np.seterr(all="raise")
                    x = x + dxdt(np.abs(V), x, Ap, An, Vp,
                                 Vn,
                                 xp, xn, alphap, alphan, 1) * dt
                    # Clip the value of state variables beyond the [0,1] range
                    x = np.clip(x, 0, 1)

                    return x

                self.x_pos = yakopcic_one_step(np.where(V > 0, V, 0), self.x_pos, self.Ap_pos, self.An_pos, self.Vp_pos,
                                               self.Vn_pos,
                                               self.alphap_pos, self.alphan_pos, self.xp_pos, self.xn_pos)
                self.x_neg = yakopcic_one_step(np.where(V < 0, np.abs(V), 0), self.x_neg, self.Ap_neg, self.An_neg,
                                               self.Vp_neg,
                                               self.Vn_neg,
                                               self.alphap_neg, self.alphan_neg, self.xp_neg, self.xn_neg)

                # Calculate the current and the resistance for the devices
                V_read = -1
                i_pos = current(V_read, self.x_pos, self.gmax_p_pos, self.bmax_p_pos,
                                self.gmax_n_pos, self.bmax_n_pos, self.gmin_p_pos, self.bmin_p_pos, self.gmin_n_pos,
                                self.bmin_n_pos)
                i_neg = current(V_read, self.x_neg, self.gmax_p_neg, self.bmax_p_neg,
                                self.gmax_n_neg, self.bmax_n_neg, self.gmin_p_neg, self.bmin_n_neg, self.gmin_n_neg,
                                self.bmin_n_neg)

                r_pos = np.divide(V_read, i_pos, out=np.zeros(V.shape, dtype=float), where=i_pos != 0)
                r_neg = np.divide(V_read, i_neg, out=np.zeros(V.shape, dtype=float), where=i_neg != 0)

                # update the two memristor pairs
                pos_memristors[:] = r_pos
                neg_memristors[:] = r_neg

                # update network weights
                # TODO it looks like neg_memristors in having no impact on the weights (but learning works so maybe it's just my impression!)
                weights[:] = gain * (1 / pos_memristors - 1 / neg_memristors)
                pass

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
import json


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

    An_pos, Ap_pos, \
    Vn_pos, Vp_pos, \
    alphan_pos, alphap_pos, \
    bmax_n_pos, bmax_p_pos, \
    bmin_n_pos, bmin_p_pos, \
    gmax_n_pos, gmax_p_pos, \
    gmin_n_pos, gmin_p_pos, \
    xn_pos, xp_pos, \
    x0_pos = initialise_yakopcic_model(mpes.noise_percentage, encoders, acts, mpes.seed)

    An_neg, Ap_neg, \
    Vn_neg, Vp_neg, \
    alphan_neg, alphap_neg, \
    bmax_n_neg, bmax_p_neg, \
    bmin_n_neg, bmin_p_neg, \
    gmax_n_neg, gmax_p_neg, \
    gmin_n_neg, gmin_p_neg, \
    xn_neg, xp_neg, \
    x0_neg = initialise_yakopcic_model(mpes.noise_percentage, encoders, acts, mpes.seed)

    pos_memristors = initialise_yakopcic_memristors(x0_pos, bmax_n_pos, bmax_p_pos,
                                                    bmin_n_pos, bmin_p_pos,
                                                    gmax_n_pos, gmax_p_pos,
                                                    gmin_n_pos, gmin_p_pos,
                                                    acts.shape[0], encoders.shape[0],
                                                    f"{rule}:pos_memristors")
    neg_memristors = initialise_yakopcic_memristors(x0_neg, bmax_n_neg, bmax_p_neg,
                                                    bmin_n_neg, bmin_p_neg,
                                                    gmax_n_neg, gmax_p_neg,
                                                    gmin_n_neg, gmin_p_neg,
                                                    acts.shape[0], encoders.shape[0],
                                                    f"{rule}:neg_memristors")

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

    model.operators.append(
        SimmPES(acts,
                local_error,
                model.sig[conn]["pos_memristors"],
                model.sig[conn]["neg_memristors"],
                model.sig[conn]["weights"],
                mpes.gain,
                bmax_n_pos,
                bmax_p_pos,
                bmin_n_pos,
                bmin_p_pos,
                gmax_n_pos,
                gmax_p_pos,
                gmin_n_pos,
                gmin_p_pos,
                Vn_pos,
                Vp_pos,
                alphan_pos,
                alphap_pos,
                An_pos,
                Ap_pos,
                x0_pos,
                xn_pos,
                xp_pos,
                bmax_n_neg,
                bmax_p_neg,
                bmin_n_neg,
                bmin_p_neg,
                gmax_n_neg,
                gmax_p_neg,
                gmin_n_neg,
                gmin_p_neg,
                Vn_neg,
                Vp_neg,
                alphan_neg,
                alphap_neg,
                An_neg,
                Ap_neg,
                x0_neg,
                xn_neg,
                xp_neg,
                mpes.initial_state,
                )
    )

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts
    model.sig[rule]["pos_memristors"] = pos_memristors
    model.sig[rule]["neg_memristors"] = neg_memristors
