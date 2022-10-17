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
    probeable = ("error", "activities", "delta", "x_pos", "x_neg")

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    gain = NumberParam("gain", readonly=True, default=5e6)
    initial_state = DictParam("initial_state", optional=True)

    def __init__(self,
                 pre_synapse=Default,
                 noisy=False,
                 gain=Default,
                 initial_state=None,
                 seed=None):
        super().__init__(size_in="post_state")

        self.pre_synapse = pre_synapse
        if not noisy:
            self.noise_percentage = 0
        elif isinstance(noisy, float) or isinstance(noisy, int):
            self.noise_percentage = noisy
        else:
            raise ValueError(f"Noisy parameter must be float or an int, not {type(noisy)}")
        self.gain = gain
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
            x_pos,
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
            x_neg,
            xn_neg,
            xp_neg,
            initial_state,
            tag=None
    ):
        super(SimmPES, self).__init__(tag=tag)

        self.gain = gain
        self.error_threshold = 1e-5
        self.readV = -1

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

        self.currents = []
        self.xs = []
        self.rs = []
        self.initial_state = initial_state

        self.pos_pulse_counter = np.zeros_like(An_pos)
        self.neg_pulse_counter = np.zeros_like(An_neg)
        self.pulse_archive = []

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error]
        self.updates = [weights, x_pos, x_neg]

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
    def x_pos(self):
        return self.updates[1]

    @property
    def x_neg(self):
        return self.updates[2]

    def _descstr(self):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.weights)

    def compute_current(self, x_pos, x_neg):
        i_pos = current(self.readV, x_pos, self.gmax_p_pos, self.bmax_p_pos,
                        self.gmax_n_pos, self.bmax_n_pos, self.gmin_p_pos, self.bmin_p_pos, self.gmin_n_pos,
                        self.bmin_n_pos)
        i_neg = current(self.readV, x_neg, self.gmax_p_neg, self.bmax_p_neg,
                        self.gmax_n_neg, self.bmax_n_neg, self.gmin_p_neg, self.bmin_p_neg, self.gmin_n_neg,
                        self.bmin_n_neg)

        return i_pos, i_neg

    def compute_resistance(self, x_pos, x_neg):
        i_pos = current(self.readV, x_pos, self.gmax_p_pos, self.bmax_p_pos,
                        self.gmax_n_pos, self.bmax_n_pos, self.gmin_p_pos, self.bmin_p_pos, self.gmin_n_pos,
                        self.bmin_n_pos)
        i_neg = current(self.readV, x_neg, self.gmax_p_neg, self.bmax_p_neg,
                        self.gmax_n_neg, self.bmax_n_neg, self.gmin_p_neg, self.bmin_p_neg, self.gmin_n_neg,
                        self.bmin_n_neg)

        return self.readV / i_pos, self.readV / i_neg

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        local_error = signals[self.error]

        x_pos = signals[self.x_pos]
        x_neg = signals[self.x_neg]
        weights = signals[self.weights]

        gain = self.gain
        error_threshold = self.error_threshold

        # ---- voltages found by pulse_experiment_1s_to_1ms.py
        readV = -1
        setV = 3.86621037038006
        resetV = -8.135891404816215
        print('readV', readV)
        print("setV: ", setV)
        print("resetV: ", resetV)

        # overwrite initial transform with memristor-based weights
        if "weights" in self.initial_state:
            weights[:] = self.initial_state["weights"]
        else:
            i_pos = current(readV, x_pos, self.gmax_p_pos, self.bmax_p_pos,
                            self.gmax_n_pos, self.bmax_n_pos, self.gmin_p_pos, self.bmin_p_pos, self.gmin_n_pos,
                            self.bmin_n_pos)
            i_neg = current(readV, x_neg, self.gmax_p_neg, self.bmax_p_neg,
                            self.gmax_n_neg, self.bmax_n_neg, self.gmin_p_neg, self.bmin_p_neg, self.gmin_n_neg,
                            self.bmin_n_neg)

            weights[:] = gain * (i_pos / readV - i_neg / readV)

        def step_simmpes():
            # set update to zero if error is small or adjustments go on forever
            # if error is small return zero delta
            if np.any(np.absolute(local_error) > error_threshold):
                # calculate the magnitude of the update based on PES learning rule
                pes_delta = np.outer(-local_error, pre_filtered)

                # some memristors are adjusted erroneously if we don't filter
                spiked_map = find_spikes(pre_filtered, weights.shape, invert=True)
                pes_delta[spiked_map] = 0

                # V = np.sign(pes_delta) * 3.86621037038006

                # V = np.select([pes_delta > 0, pes_delta < 0], [setV, resetV])

                # -- count pulses given to each memristor for post-hoc statistics
                pos_temp = np.sign(pes_delta)
                pos_temp[pos_temp < 0] = 0
                neg_temp = np.sign(pes_delta)
                neg_temp[neg_temp > 0] = 0
                neg_temp = np.abs(neg_temp)
                self.pos_pulse_counter += pos_temp
                self.neg_pulse_counter += neg_temp
                self.pulse_archive.append(np.sign(pes_delta))

                def yakopcic_one_step(V, x, Ap, An, Vp, Vn, alphap, alphan, xp, xn):
                    # Calculate the state variables at the current timestep
                    np.seterr(all="raise")
                    x = x + dxdt(V, x, Ap, An, Vp,
                                 Vn,
                                 xp, xn, alphap, alphan, 1) * dt
                    # Clip the value of state variables beyond the [0,1] range
                    x = np.clip(x, 0, 1)

                    return x

                x_pos_old = copy(x_pos)
                x_neg_old = copy(x_neg)
                # -- update memristor states
                update_direction = np.sign(pes_delta)
                # ---- POTENTIATE case
                mask_potentiate = update_direction > 0
                x_pos[mask_potentiate] = yakopcic_one_step(
                    setV * np.ones_like(x_pos[mask_potentiate]),
                    x_pos[mask_potentiate],
                    self.Ap_pos[mask_potentiate], self.An_pos[mask_potentiate],
                    self.Vp_pos[mask_potentiate], self.Vn_pos[mask_potentiate],
                    self.alphap_pos[mask_potentiate], self.alphan_pos[mask_potentiate],
                    self.xp_pos[mask_potentiate], self.xn_pos[mask_potentiate])
                x_neg[mask_potentiate] = yakopcic_one_step(
                    resetV * np.ones_like(x_neg[mask_potentiate]),
                    x_neg[mask_potentiate],
                    self.Ap_neg[mask_potentiate], self.An_neg[mask_potentiate],
                    self.Vp_neg[mask_potentiate], self.Vn_neg[mask_potentiate],
                    self.alphap_neg[mask_potentiate], self.alphan_neg[mask_potentiate],
                    self.xp_neg[mask_potentiate], self.xn_neg[mask_potentiate])
                # ---- DEPRESS case
                mask_depress = update_direction < 0
                x_pos[mask_depress] = yakopcic_one_step(
                    resetV * np.ones_like(x_pos[mask_depress]),
                    x_pos[mask_depress],
                    self.Ap_pos[mask_depress], self.An_pos[mask_depress],
                    self.Vp_pos[mask_depress], self.Vn_pos[mask_depress],
                    self.alphap_pos[mask_depress], self.alphan_pos[mask_depress],
                    self.xp_pos[mask_depress], self.xn_pos[mask_depress])
                x_neg[mask_depress] = yakopcic_one_step(
                    setV * np.ones_like(x_neg[mask_depress]),
                    x_neg[mask_depress],
                    self.Ap_neg[mask_depress], self.An_neg[mask_depress],
                    self.Vp_neg[mask_depress], self.Vn_neg[mask_depress],
                    self.alphap_neg[mask_depress], self.alphan_neg[mask_depress],
                    self.xp_neg[mask_depress], self.xn_neg[mask_depress])

                diff_pos = np.sign(x_pos - x_pos_old)
                diff_neg = np.sign(x_neg - x_neg_old)

                # -- calculate the current through the devices
                i_pos = current(readV, x_pos, self.gmax_p_pos, self.bmax_p_pos,
                                self.gmax_n_pos, self.bmax_n_pos, self.gmin_p_pos, self.bmin_p_pos, self.gmin_n_pos,
                                self.bmin_n_pos)
                i_neg = current(readV, x_neg, self.gmax_p_neg, self.bmax_p_neg,
                                self.gmax_n_neg, self.bmax_n_neg, self.gmin_p_neg, self.bmin_p_neg, self.gmin_n_neg,
                                self.bmin_n_neg)

                # -- update network weights using the reciprocal of Ohm's Law G = I / V (R = V / I)
                weights[:] = gain * (i_pos / readV - i_neg / readV)

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

    # -- Instantiate two sets of Yakopcic memristors
    An_pos, Ap_pos, \
    Vn_pos, Vp_pos, \
    alphan_pos, alphap_pos, \
    bmax_n_pos, bmax_p_pos, \
    bmin_n_pos, bmin_p_pos, \
    gmax_n_pos, gmax_p_pos, \
    gmin_n_pos, gmin_p_pos, \
    xn_pos, xp_pos, \
    x0_pos = initialise_yakopcic_model(mpes.noise_percentage, encoders, acts,
                                       mpes.seed)

    An_neg, Ap_neg, \
    Vn_neg, Vp_neg, \
    alphan_neg, alphap_neg, \
    bmax_n_neg, bmax_p_neg, \
    bmin_n_neg, bmin_p_neg, \
    gmax_n_neg, gmax_p_neg, \
    gmin_n_neg, gmin_p_neg, \
    xn_neg, xp_neg, \
    x0_neg = initialise_yakopcic_model(mpes.noise_percentage, encoders, acts,
                                       mpes.seed + 1 if mpes.seed is not None else None)

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

    x_pos = Signal(shape=(encoders.shape[0], acts.shape[0]), name=f"{rule}:x_pos",
                   initial_value=x0_pos)
    x_neg = Signal(shape=(encoders.shape[0], acts.shape[0]), name=f"{rule}:x_neg",
                   initial_value=x0_neg)

    model.operators.append(
        SimmPES(acts, local_error, model.sig[conn]["weights"], mpes.gain, bmax_n_pos, bmax_p_pos, bmin_n_pos,
                bmin_p_pos, gmax_n_pos, gmax_p_pos, gmin_n_pos, gmin_p_pos, Vn_pos, Vp_pos, alphan_pos, alphap_pos,
                An_pos, Ap_pos, x_pos, xn_pos, xp_pos, bmax_n_neg, bmax_p_neg, bmin_n_neg, bmin_p_neg, gmax_n_neg,
                gmax_p_neg, gmin_n_neg, gmin_p_neg, Vn_neg, Vp_neg, alphan_neg, alphap_neg, An_neg, Ap_neg, x_neg,
                xn_neg, xp_neg, mpes.initial_state)
    )

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts
    model.sig[rule]['x_pos'] = x_pos
    model.sig[rule]['x_neg'] = x_neg
