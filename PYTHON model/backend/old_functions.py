from yakopcic_model import *
from yakopcic_functions import *
import scipy.signal


def plot_images(plot_type, time, voltage, i, r, x, model=None):
    peak_ids = scipy.signal.find_peaks(r)
    peak = [0]
    for idx in peak_ids[0]:
        peak.append(r[idx])

    if plot_type == 1:  # Plots regular resistance plot; Full plot + its local peaks.
        fig_plot, ax_plot = plt.subplots(2, 1, figsize=(7, 5))

        ax_plot[0].plot(time, r)  # Solution otherwise.
        ax_plot[0].twinx().plot(time, voltage, color='r')  # Voltage.
        ax_plot[0].set_yscale("log")
        ax_plot[1].plot(peak, "o")
        ax_plot[1].set_yscale("log")
        fig_plot.tight_layout()

        fig_plot.show()

    else:  # Plots the IV curve.
        fig_plot, ax_plot = plt.subplots(2, 1, figsize=(7, 5))
        ax_plot[0].plot(time, i)
        ax_plot[0].twinx().plot(time, voltage, color='r')
        ax_plot[1].plot(voltage, i)
        # ax_plot[1].set_ylim(-0.006, 0.006)
        fig_plot.tight_layout()

        fig_plot.show()

    if model is not None:  # Debug plot, shows all the relevant parameters (V, I, x, g, f).
        fig_debug, ax_debug = plt.subplots(5, 1, figsize=(12, 10))

        ax_debug[0].plot(time, voltage)
        ax_debug[0].set_ylabel("Voltage")
        ax_debug[1].plot(time, i)
        ax_debug[1].set_ylabel("Current")
        ax_debug[2].plot(time, x)
        ax_debug[2].set_ylabel("State Variable")
        ax_debug[3].plot(time, g(voltage, model['Ap'], model['An'], model['Vp'], model['An']))
        ax_debug[3].set_ylabel("g")
        ax_debug[4].plot(time, f(voltage, x, model['xp'], model['xn'], model['alphap'], model['alphan'], model['eta']))
        ax_debug[4].set_ylabel("f")

        for ax in ax_debug.ravel():
            ax.set_xlabel("Time")

        fig_debug.tight_layout()
        fig_debug.show()

    plt.show()
    return fig_plot

def startup2(lines):
    iptVs = {}
    lines = lines.split('\n')
    wave_number = 1
    for line in lines:
        t_rise, t_on, t_fall, t_off, V_on, V_off, n_cycles = map(float, line.split())
        iptV = {"t_rise": t_rise, "t_on": t_on, "t_fall": t_fall, "t_off": t_off, "V_on": V_on, "V_off": V_off,
                "n_cycles": int(n_cycles)}
        iptVs["{}".format(wave_number)] = iptV
        wave_number += 1

    return iptVs


def interactive_iv(iptVs, dt):
    t = 0
    print("dt: ", dt)
    for iptV in iptVs.values():
        for j in range(0, int(iptV['n_cycles'])):
            if j == 0 and iptVs["1"] == iptV:
                t, v_total = generate_wave(iptV, dt, t)
            else:
                t, v_total = generate_wave(iptV, dt, t, v_total)

    time = np.linspace(0, t, len(v_total))
    return time, v_total


def generate_wave(iv, dt, t, base=None):
    print(iv["t_rise"], iv["t_fall"], iv["t_on"], iv["t_off"])
    t += (iv["t_rise"] + iv["t_fall"] + iv["t_on"] + iv["t_off"])
    v1 = np.linspace(iv["V_off"], iv["V_on"], round(iv["t_rise"] * 1 / dt))
    v2 = iv["V_on"] * np.ones(round(iv["t_on"] * 1 / dt))
    v3 = np.linspace(iv["V_on"], iv["V_off"], round(iv["t_fall"] * 1 / dt))
    v4 = np.array([]) if iv["t_off"] == 0 else iv["V_off"] * np.ones(round(iv["t_off"] * 1 / dt))
    vtotal = np.concatenate((base, v1, v2, v3, v4)) if base is not None else np.concatenate((v1, v2, v3, v4))
    return t, vtotal