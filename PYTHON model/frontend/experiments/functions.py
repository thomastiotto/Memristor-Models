from yakopcic_model import *
from yakopcic_functions import *
import scipy.signal
import matplotlib.ticker as mticker


def set_pulse(resetV, setV, pulse_length):
    print('------------------')
    print('Pulse length:', pulse_length, 's')
    print('RESET:', resetV, 'V')
    print('SET:', setV, 'V')

    # FORMAT
    # "t_rise", "t_on":, "t_fall", "t_off", "V_on", "V_off", "n_cycles"
    return f""".001 120 .001 .01 1 0 1
.001 {pulse_length} .001 .4 {resetV} -.1 10   
.001 {pulse_length} .001 .4 {setV} -.1 10"""


def find_peaks(r, dt=0.001, consider_from=120, ipd=None, num_peaks=20, debug=False):
    consider_from = int(consider_from / dt)
    r_ranged = r[consider_from:]

    peaks = [[0]]
    count = 0
    while len(peaks[0]) < num_peaks:
        peaks = scipy.signal.find_peaks(r_ranged, distance=(ipd / dt) - (count * (ipd / dt) / 10))

        if debug:
            plt.show()
            plt.title(f'Peaks found: {len(peaks[0])}, IPD: {(ipd / dt) - (count * (ipd / dt) / 10)}')
            plt.plot(r_ranged)
            plt.plot(peaks[0], r_ranged[peaks[0]], 'o')
            plt.show()

        count += 1

    return r_ranged[peaks[0]]


def plot_images(time, voltage, i, r, x, ipd, num_peaks, label, fig=None, plot_type='pulse', debug=None, model=None):
    peaks = find_peaks(r, ipd=ipd, num_peaks=num_peaks)

    if plot_type == 'pulse':  # Plots regular resistance plot; Full plot + its local peaks.
        # w, h = plt.figaspect(1 / 3.5)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h), dpi=300)
        if fig is not None:
            fig_plot = fig
            ax_plot = fig.axes
        else:
            fig_plot, ax_plot = plt.subplots(2, 1, figsize=(12, 10))

        ax_plot[0].plot(time, r, label=label)  # Solution otherwise.
        ax_plot[0].twinx().plot(time, voltage, color='r', label='Voltage')  # Voltage.
        ax_plot[0].set_yscale("log")
        ax_plot[0].set_title("Experiment pulses")
        ax_plot[0].set_xlabel("Time (s)")
        ax_plot[0].set_ylabel("Resistance (Ohm)")
        ax_plot[0].legend(loc='best')

        ax_plot[1].plot(peaks, "o", fillstyle='none', label=label)
        ax_plot[1].xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_plot[1].set_yscale("log")
        ax_plot[1].set_xlabel("Pulse number")
        ax_plot[1].set_ylabel("Resistance (Ohm)")
        ax_plot[1].set_title("Resistance after 120 s SET")
        ax_plot[1].legend(loc='best')

        fig_plot.tight_layout()
    elif plot_type == 'iv':  # Plots the IV curve.
        fig_plot, ax_plot = plt.subplots(2, 1, figsize=(7, 5))
        ax_plot[0].plot(time, i)
        ax_plot[0].twinx().plot(time, voltage, color='r')
        ax_plot[1].plot(voltage, i)
        # ax_plot[1].set_ylim(-0.006, 0.006)

        fig_plot.tight_layout()
    else:
        return None, None

    fig_plot = (fig_plot, None)

    if debug and model:  # Debug plot, shows all the relevant parameters (V, I, x, g, f).
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

        fig_plot = (fig_plot, fig_debug)

    return fig_plot


def startup2(lines):
    iptVs = {}
    lines = lines.split('\n')

    ipds = []
    num_waves = []
    wave_number = 1
    for line in lines:
        t_rise, t_on, t_fall, t_off, V_on, V_off, n_cycles = map(float, line.split())
        iptV = {"t_rise": t_rise, "t_on": t_on, "t_fall": t_fall, "t_off": t_off, "V_on": V_on, "V_off": V_off,
                "n_cycles": int(n_cycles)}
        iptVs["{}".format(wave_number)] = iptV
        wave_number += 1

        num_waves.append(n_cycles)
        ipds.append(t_rise + t_on + t_fall + t_off)

    return iptVs, ipds, num_waves


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
