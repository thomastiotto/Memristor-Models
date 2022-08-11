from yakopcic_model import *
import yakopcic_functions
import scipy.signal

# Euler step-based solver that calculates the state variable and current for each time point.
# Return the resulting two arrays.
def solver2(f, time, dt, iv, v, args=[]):
    x_sol = np.zeros(len(time))
    x_sol[0] = iv

    for i in range(1, len(time)):
        x = euler_step(x_sol[i-1], time[i], f, dt, v[i], args)
        if x < 0:
            x = 0
        if x > 1:
            x = 1

        x_sol[i] = x

    return x_sol


# Produces the voltage pulses based on the given inputs.
def interactive_iv(iptVs, dt):
    t = 0
    print("dt: ", dt)
    for iptV in iptVs.values():
        for j in range(0, int(iptV['n_cycles'])):
            if j == 0 and iptVs["1"] == iptV:
                t, v_total = generate_wave(iptV, dt, t)
            else:
                t, v_total = generate_wave(iptV, dt, t, v_total)

    time = np.linspace(0, t+dt, len(v_total))
    return time, v_total


def generate_wave(iv, dt, t, base=None):
    base = np.array([0]) if base is None else base
    t += (iv["t_rise"] + iv["t_fall"] + iv["t_on"] + iv["t_off"])
    v1 = np.linspace(iv["V_off"], iv["V_on"], round(iv["t_rise"] * 1/dt))
    v2 = iv["V_on"] * np.ones(round(iv["t_on"] * 1/dt))
    v3 = np.linspace(iv["V_on"], iv["V_off"], round(iv["t_fall"] * 1/dt))
    v4 = np.array([]) if iv["t_off"] == 0 else iv["V_off"] * np.ones(round(iv["t_off"] * 1/dt))
    vtotal = np.concatenate((base, v1, v2, v3, v4))
    return t, vtotal


# Finds the indices that are representing local peaks.
# Then, make a list of local peaks using the incides.
# Finally, plots the graphs.
def plot_images(filename, plot_type, time, voltage, i, r, x, memr=None):
    peak_ids = scipy.signal.find_peaks(r)
    peak = [0]
    for idx in peak_ids[0]:
        peak.append(r[idx])

    if plot_type == 1:  # Plots regular resistance plot; Full plot + its local peaks.
        fig_plot, ax_plot = plt.subplots(2, 1, figsize=(7, 5))
        if filename == "input.txt":
            ax_plot[0].plot(time[120000:], r[120000:])  # Supposes a 120s SET pulse!
            ax_plot[0].twinx().plot(time[120000:], voltage[120000:], color='r')  # Voltage.
        else:
            ax_plot[0].plot(time, r)  # Solution otherwise.
            ax_plot[0].twinx().plot(time, voltage, color='r')  # Voltage.
        ax_plot[0].set_yscale("log")
        ax_plot[1].plot(peak, "o")
        ax_plot[1].set_yscale("log")
        fig_plot.tight_layout()
        fig_plot.show()

    else:  # Plots the IV curve.
        fig_plot, ax_plot = plt.subplots(2, 1, figsize=(7, 5))
        ax_plot[0].plot(time, r)
        ax_plot[0].twinx().plot(time, voltage, color='r')
        ax_plot[1].plot(voltage, i)
        ax_plot[1].set_ylim(-0.006, 0.006)
        fig_plot.tight_layout()
        fig_plot.show()

    if memr is not None: # Debug plot, shows all the relevant parameters (V, I, x, g, f).
        fig_debug, ax_debug = plt.subplots(5, 1, figsize=(12, 10))
        ax_debug[0].plot(time, voltage)
        ax_debug[0].set_ylabel("Voltage")
        ax_debug[1].plot(time, i)
        ax_debug[1].set_ylabel("Current")
        ax_debug[2].plot(time, x)
        ax_debug[2].set_ylabel("State Variable")
        ax_debug[3].plot(time, yakopcic_functions.g(voltage, memr.Ap, memr.An, memr.Vp, memr.An))
        ax_debug[3].set_ylabel("g")
        ax_debug[4].plot(time, yakopcic_functions.f(voltage, x, memr.xp, memr.xn, memr.alphap, memr.alphan, memr.eta))
        ax_debug[4].set_ylabel("f")

        for ax in ax_debug.ravel():
            ax.set_xlabel("Time")
        fig_debug.tight_layout()
        fig_debug.show()
    plt.show()

