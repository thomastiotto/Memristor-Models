import warnings
from copy import copy
from random import random

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
    yakopcic_model = json.load(open('../../fitted/fitting_pulses/new_device/mystery_model'))

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
    x0 = get_truncated_normal(0.0032239748515913717, 0.0032239748515913717 * noise_percentage,
                              0, 1,
                              encoders.shape[0], acts.shape[0])
    dt = yakopcic_model['dt']

    return An, Ap, Vn, Vp, alphan, alphap, bmax_n, bmax_p, bmin_n, bmin_p, gmax_n, gmax_p, gmin_n, gmin_p, xn, xp, x0, dt


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
                 seed=None,
                 strategy='symmetric-probabilistic',
                 setP=1,
                 resetP=1,
                 # voltages found in NL_characterisation.py, these can be used with P(SET)=P(RESET)
                 resetV=-6.4688295585009605,
                 setV=0.24694177778629942,
                 high_precision=True,
                 program_length=7,
                 read_length=3):
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

        if strategy == 'symmetric-probabilistic':
            if setP is None or resetP is None:
                raise ValueError(
                    ", setP and resetP must be set for symmetric-probabilistic strategy")
            self.setP = setP
            self.resetP = resetP
        elif strategy == 'asymmetric-probabilistic':
            if resetP is None:
                raise ValueError("resetP must be specified for asymmetric-probabilistic strategy")
            self.setP = 1 - resetP
            self.resetP = resetP
        elif strategy == 'symmetric':
            self.setP = 1.0
            self.resetP = 1.0
        elif strategy == 'asymmetric':
            self.setP = 1.0
            self.resetP = 0.0
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        self.setV = setV
        self.resetV = resetV
        self.high_precision = high_precision
        self.program_length = program_length
        self.read_length = read_length

        print(f'Using {strategy} strategy: P(SET)={self.setP}, P(RESET)={self.resetP}')
        print(f'Voltage amplitudes: setV={self.setV} V, resetV={self.resetV} V')
        print('High' if self.high_precision else 'Low', 'precision mode')


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
            setP,
            resetP,
            setV,
            resetV,
            dt,
            high_precision,
            program_length,
            read_length,
            tag=None
    ):
        super(SimmPES, self).__init__(tag=tag)

        self.gain = gain
        self.error_threshold = 1e-5
        self.readV = -1
        self.setP = setP
        self.resetP = resetP
        self.setV = setV
        self.resetV = resetV

        self.dt = dt
        self.high_precision = high_precision
        self.program_length = program_length
        self.read_length = read_length

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

        self.pos_pulse_archive = []
        self.neg_pulse_archive = []

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

        readV = -0.5

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
            def yakopcic_one_step(V, x, Ap, An, Vp, Vn, alphap, alphan, xp, xn):
                # Calculate the state variables at the current timestep
                np.seterr(all="raise")
                x = x + dxdt(V, x, Ap, An, Vp,
                             Vn,
                             xp, xn, alphap, alphan, 1) * self.dt
                # Clip the value of state variables beyond the [0,1] range
                x = np.clip(x, 0, 1)

                return x

            def yakopcic_update(V, x, steps, Ap, An, Vp, Vn, alphap, alphan, xp, xn):
                for _ in range(steps):
                    x = yakopcic_one_step(V, x, Ap, An, Vp, Vn, alphap, alphan, xp, xn)

                return x

            # set update to zero if error is small or adjustments go on forever
            # if error is small return zero delta
            if np.any(np.absolute(local_error) > error_threshold):
                # calculate the magnitude of the update based on PES learning rule
                pes_delta = np.outer(-local_error, pre_filtered)

                # some memristors are adjusted erroneously if we don't filter
                spiked_map = find_spikes(pre_filtered, weights.shape, invert=True)
                pes_delta[spiked_map] = 0

                # -- update memristor states
                update_direction = np.sign(pes_delta)

                # compute SET and RESET probabilities for each synapse
                device_selection_set = np.random.rand(*update_direction.shape)
                device_selection_reset = np.random.rand(*update_direction.shape)

                # POTENTIATE
                mask_potentiate_set = np.logical_and(update_direction > 0,
                                                     device_selection_set < self.setP)
                x_pos[mask_potentiate_set] = yakopcic_update(
                    self.setV * np.ones_like(x_pos[mask_potentiate_set]),
                    x_pos[mask_potentiate_set],
                    self.program_length,
                    self.Ap_pos[mask_potentiate_set],
                    self.An_pos[mask_potentiate_set],
                    self.Vp_pos[mask_potentiate_set],
                    self.Vn_pos[mask_potentiate_set],
                    self.alphap_pos[mask_potentiate_set],
                    self.alphan_pos[mask_potentiate_set],
                    self.xp_pos[mask_potentiate_set],
                    self.xn_pos[mask_potentiate_set])
                mask_potentiate_reset = np.logical_and(update_direction > 0,
                                                       device_selection_reset < self.resetP)
                x_neg[mask_potentiate_reset] = yakopcic_update(
                    self.resetV * np.ones_like(x_neg[mask_potentiate_reset]),
                    x_neg[mask_potentiate_reset],
                    self.program_length,
                    self.Ap_neg[mask_potentiate_reset],
                    self.An_neg[mask_potentiate_reset],
                    self.Vp_neg[mask_potentiate_reset],
                    self.Vn_neg[mask_potentiate_reset],
                    self.alphap_neg[mask_potentiate_reset],
                    self.alphan_neg[mask_potentiate_reset],
                    self.xp_neg[mask_potentiate_reset],
                    self.xn_neg[mask_potentiate_reset])
                # DEPRESS
                mask_depress_set = np.logical_and(update_direction < 0,
                                                  device_selection_set < self.setP)
                x_neg[mask_depress_set] = yakopcic_update(
                    self.setV * np.ones_like(x_neg[mask_depress_set]),
                    x_neg[mask_depress_set],
                    self.program_length,
                    self.Ap_neg[mask_depress_set],
                    self.An_neg[mask_depress_set],
                    self.Vp_neg[mask_depress_set],
                    self.Vn_neg[mask_depress_set],
                    self.alphap_neg[mask_depress_set],
                    self.alphan_neg[mask_depress_set],
                    self.xp_neg[mask_depress_set],
                    self.xn_neg[mask_depress_set])
                mask_depress_reset = np.logical_and(update_direction < 0,
                                                    device_selection_reset < self.resetP)
                x_pos[mask_depress_reset] = yakopcic_update(
                    self.resetV * np.ones_like(x_pos[mask_depress_reset]),
                    x_pos[mask_depress_reset],
                    self.program_length,
                    self.Ap_pos[mask_depress_reset],
                    self.An_pos[mask_depress_reset],
                    self.Vp_pos[mask_depress_reset],
                    self.Vn_pos[mask_depress_reset],
                    self.alphap_pos[mask_depress_reset],
                    self.alphan_pos[mask_depress_reset],
                    self.xp_pos[mask_depress_reset],
                    self.xn_pos[mask_depress_reset])

                ts_pos_pulses = mask_potentiate_set.astype(int)
                ts_neg_pulses = -1 * mask_potentiate_reset.astype(int)
                ts_neg_pulses = ts_neg_pulses + mask_depress_set.astype(int)
                ts_pos_pulses = ts_pos_pulses + -1 * mask_depress_reset.astype(int)
                self.pos_pulse_archive.append(ts_pos_pulses)
                self.neg_pulse_archive.append(ts_neg_pulses)

            # -- reading cycle
            x_pos[:] = yakopcic_update(
                self.readV * np.ones_like(x_pos),
                x_pos,
                self.read_length,
                self.Ap_pos,
                self.An_pos,
                self.Vp_pos,
                self.Vn_pos,
                self.alphap_pos,
                self.alphan_pos,
                self.xp_pos,
                self.xn_pos)

            x_neg[:] = yakopcic_update(
                self.readV * np.ones_like(x_neg),
                x_neg,
                self.read_length,
                self.Ap_neg,
                self.An_neg,
                self.Vp_neg,
                self.Vn_neg,
                self.alphap_neg,
                self.alphan_neg,
                self.xp_neg,
                self.xn_neg)

            # -- calculate the current through the devices
            # ---- (there'll be a slight mismatch because in find_peaks() we looked at the centre of the read interval
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
        x0_pos, dt_yk = initialise_yakopcic_model(mpes.noise_percentage, encoders, acts,
                                                  mpes.seed)

    An_neg, Ap_neg, \
        Vn_neg, Vp_neg, \
        alphan_neg, alphap_neg, \
        bmax_n_neg, bmax_p_neg, \
        bmin_n_neg, bmin_p_neg, \
        gmax_n_neg, gmax_p_neg, \
        gmin_n_neg, gmin_p_neg, \
        xn_neg, xp_neg, \
        x0_neg, dt_yk = initialise_yakopcic_model(mpes.noise_percentage, encoders, acts,
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

    if mpes.high_precision:
        program_length = mpes.program_length
        read_length = mpes.read_length
        dt = dt_yk
    else:
        program_length = 1
        read_length = 1
        dt = mpes.program_length * dt_yk + mpes.read_length * dt_yk

    model.operators.append(
        SimmPES(acts, local_error, model.sig[conn]["weights"], mpes.gain, bmax_n_pos, bmax_p_pos, bmin_n_pos,
                bmin_p_pos, gmax_n_pos, gmax_p_pos, gmin_n_pos, gmin_p_pos, Vn_pos, Vp_pos, alphan_pos, alphap_pos,
                An_pos, Ap_pos, x_pos, xn_pos, xp_pos, bmax_n_neg, bmax_p_neg, bmin_n_neg, bmin_p_neg, gmax_n_neg,
                gmax_p_neg, gmin_n_neg, gmin_p_neg, Vn_neg, Vp_neg, alphan_neg, alphap_neg, An_neg, Ap_neg, x_neg,
                xn_neg, xp_neg, mpes.initial_state, mpes.setP, mpes.resetP, mpes.setV, mpes.resetV, dt,
                mpes.high_precision, program_length, read_length)
    )

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts
    model.sig[rule]['x_pos'] = x_pos
    model.sig[rule]['x_neg'] = x_neg
