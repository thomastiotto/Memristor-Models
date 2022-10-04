from yakopcic_model import *
from yakopcic_functions import *
import scipy.signal
import matplotlib.ticker as mticker


def set_pulse(resetV, setV, pulse_length, readV, read_length):
    print('------------------')
    print('RESET:', resetV, 'V')
    print('SET:', setV, 'V')
    print('Pulse length:', pulse_length, 's')
    print('READ:', readV, 'V')
    print('READ length:', read_length, 's')

    # FORMAT
    # "t_rise", "t_on":, "t_fall", "t_off", "V_on", "V_off", "n_cycles"
    return f""".001 120 .001 .01 1 0 1
.001 {pulse_length} .001 {read_length} {resetV} {readV} 10   
.001 {pulse_length} .001 {read_length} {setV} {readV} 10"""


def find_peaks(r, voltage, readV, dt=0.001, consider_from=120, debug=False):
    consider_from = int(consider_from / dt)
    r_ranged = r[consider_from:]
    v_ranged = voltage[consider_from:]

    # -- find peaks
    data = np.where(v_ranged == readV)[0]
    diffs = np.diff(data) != 1
    indexes = np.nonzero(diffs)[0] + 1
    groups = np.split(data, indexes)
    peaks_idx = []
    for g in groups[1:]:
        interval_centr = int(g[0] + (g[-1] - g[0]) / 2)
        peaks_idx.append(interval_centr)
    peaks = r_ranged[peaks_idx]

    if debug:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.plot(v_ranged, color='r')
        ax.plot((peaks_idx,), readV, 'x', color='r')
        ax2 = ax.twinx()
        ax2.plot(r_ranged, color='b')
        ax2.plot(peaks_idx, r_ranged[peaks_idx], 'x', color='b')
        fig.suptitle(f'Peaks found: {len(peaks_idx)} at {readV} V')
        fig.show()

    return peaks


def plot_images(time, voltage, i, r, x, label, readV=None, fig=None, plot_type='pulse', show_peaks=False, model=None):
    peaks = find_peaks(r, voltage, readV, debug=show_peaks)

    if plot_type == 'pulse':  # Plots regular resistance plot; Full plot + its local peaks.
        # w, h = plt.figaspect(1 / 3.5)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h), dpi=300)
        assert readV is not None

        if fig is not None:
            fig_plot = fig
            ax_plot = fig_plot.axes
        else:
            fig_plot, ax_plot = plt.subplots(1, 1, figsize=(6, 5))

        ax_plot[0].plot(peaks, "o", fillstyle='none', label=label)
        ax_plot[0].xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_plot[0].set_yscale("log")
        ax_plot[0].set_xlabel("Pulse number")
        ax_plot[0].set_ylabel("Resistance (Ohm)")
        ax_plot[0].set_title("Resistance after 120 s SET")
        ax_plot[0].legend(loc='best')

        fig_plot.tight_layout()

        return fig_plot

    elif plot_type == 'debug':
        assert model is not None

        fig_debug, ax_debug = plt.subplots(6, 1, figsize=(12, 10))

        ax_debug[0].plot(time, voltage)
        ax_debug[0].set_ylabel("Voltage")
        ax_debug[1].plot(time, np.abs(i))
        ax_debug[1].set_ylabel("Current")
        ax_debug[2].plot(time, r)
        ax_debug[2].set_ylabel("Resistance")
        ax_debug[3].plot(time, x)
        ax_debug[3].set_ylabel("State Variable")
        ax_debug[4].plot(time, g(voltage, model['Ap'], model['An'], model['Vp'], model['An']))
        ax_debug[5].set_ylabel("g")
        ax_debug[5].plot(time, f(voltage, x, model['xp'], model['xn'], model['alphap'], model['alphan'], model['eta']))
        ax_debug[5].set_ylabel("f")

        for ax in ax_debug.ravel():
            ax.set_xlabel("Time")

        fig_debug.tight_layout()

        return fig_debug
    else:
        return None


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


# TODO return sample_at to guide sampling instants when finding peaks as the automated method is unreliable


def interactive_iv(iptVs, dt):
    t = 0
    print("dt: ", dt)
    for iptV in iptVs.values():
        for j in range(0, int(iptV['n_cycles'])):
            if j == 0 and iptVs["1"] == iptV:
                t, v_total = generate_wave(iptV, dt, t)
            else:
                t, v_total = generate_wave(iptV, dt, t, v_total)

    time = np.linspace(0, t + dt, len(v_total))
    return time, v_total


def generate_wave(iv, dt, t, base=None):
    base = np.array([0]) if base is None else base
    t += (iv["t_rise"] + iv["t_fall"] + iv["t_on"] + iv["t_off"])
    v1 = np.linspace(iv["V_off"], iv["V_on"], round(iv["t_rise"] * 1 / dt))
    v2 = iv["V_on"] * np.ones(round(iv["t_on"] * 1 / dt))
    v3 = np.linspace(iv["V_on"], iv["V_off"], round(iv["t_fall"] * 1 / dt))
    v4 = np.array([]) if iv["t_off"] == 0 else iv["V_off"] * np.ones(round(iv["t_off"] * 1 / dt))
    vtotal = np.concatenate((base, v1, v2, v3, v4))
    return t, vtotal
