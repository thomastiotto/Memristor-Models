import copy

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import ScalarFormatter

import functions
from functions import *
from models import *
from experiments import *

experiment = hp_labs_sine()

time = experiment.simulation["time"]
dt = experiment.simulation["dt"]
x0 = experiment.simulation["x0"]
dxdt = experiment.functions["dxdt"]
V = experiment.functions["V"]
I = experiment.functions["I"]

x_solve_ivp = solve_ivp(dxdt, (time[0], time[-1]), [x0], method="LSODA", t_eval=time)

t = x_solve_ivp.t
x = x_solve_ivp.y[0, :]

v = V(t)
i = I(t, x)

fig, lines, axes = plot_memristor(v, i, t, "HP Labs", figsize=(12, 6), iv_arrows=False)

################################################
#                       GUI
################################################

colour = "lightgoldenrodyellow"

ax_reset = plt.axes([0.005, 0.22, 0.1, 0.04])
button_input = Button(ax_reset, "Switch V", color=colour, hovercolor='0.975')

ax_reset = plt.axes([0.35, 0.025, 0.1, 0.04])
button_reset = Button(ax_reset, 'Reset', color=colour, hovercolor='0.975')

# create the voltage sliders
fig.subplots_adjust(left=0.15)
voltage_sliders = []
ax_vp = plt.axes([0.02, 0.3, 0.02, 0.25], facecolor=colour)
slider_vp = Slider(
        ax_vp,
        r"$V^+$",
        valmin=0,
        valmax=10,
        valinit=experiment.input_args["vp"],
        valstep=0.1,
        valfmt=r"%.2f $V$",
        orientation="vertical"
        )
voltage_sliders.append(slider_vp)
ax_vn = plt.axes([0.07, 0.3, 0.02, 0.25], facecolor=colour)
slider_vn = Slider(
        ax_vn,
        r"$V^-$",
        valmin=0,
        valmax=10,
        valinit=experiment.input_args["vn"],
        valstep=0.1,
        valfmt=r"%.2f $V$",
        orientation="vertical"
        )
voltage_sliders.append(slider_vn)
ax_f = plt.axes([0.045, 0.65, 0.02, 0.25], facecolor=colour)
slider_frequency = Slider(
        ax_f,
        r"$\nu$",
        valmin=0.1,
        valmax=100,
        valinit=experiment.input_args["frequency"],
        valstep=0.1,
        valfmt=r"%.2f $Hz$",
        orientation="vertical"
        )
voltage_sliders.append(slider_frequency)

# create experiment sliders
fig.subplots_adjust(top=0.8)
experiment_sliders = []
ax_time = plt.axes([0.15, 0.85, 0.36, 0.03], facecolor=colour)
slider_time = Slider(
        ax_time,
        r"Time",
        valmin=0,
        valmax=10,
        valstep=1,
        valinit=experiment.simulation["t_max"],
        valfmt=r"%.2f $s$"
        )
experiment_sliders.append(slider_time)

# create the memristor sliders
fig.subplots_adjust(bottom=0.3)
memristor_sliders = []
ax_d = plt.axes([0.05, 0.15, 0.25, 0.03], facecolor=colour)
slider_d = Slider(
        ax_d,
        r"$D$",
        valmin=1e-9,
        valmax=100e-9,
        valinit=experiment.memristor.D,
        valfmt=r"%.2E $m$"
        )
memristor_sliders.append(slider_d)
ax_ron = plt.axes([0.05, 0.1, 0.25, 0.03], facecolor=colour)
slider_ron = Slider(
        ax_ron,
        r"$R_{ON}$",
        valmin=1e3,
        valmax=100e3,
        valinit=experiment.memristor.RON,
        valfmt=r"%.2E $\Omega$"
        )
memristor_sliders.append(slider_ron)
ax_roff = plt.axes([0.5, 0.15, 0.25, 0.03], facecolor=colour)
slider_roff = Slider(
        ax_roff,
        r"$R_{OFF}$",
        valmin=10e3,
        valmax=1000e3,
        valinit=experiment.memristor.ROFF,
        valfmt=r"%.2E $\Omega$"
        )
memristor_sliders.append(slider_roff)
ax_mud = plt.axes([0.5, 0.1, 0.25, 0.03], facecolor=colour)
slider_mud = Slider(
        ax_mud,
        r"$\mu_D$",
        valmin=1e-15,
        valmax=10e-14,
        valinit=experiment.memristor.muD,
        valfmt=r"%.2E $m^2 s^{-1} V^{-1}$"
        )
memristor_sliders.append(slider_mud)


################################################
#                 Event handlers
################################################

def switch_input(event):
    memristor_args = [sl.val for sl in memristor_sliders]

    if experiment.input_function.shape == "sine":
        new_shape = "triangle"
    elif experiment.input_function.shape == "triangle":
        new_shape = "sine"

    experiment.input_function.shape = new_shape
    experiment.input_function.vp = slider_vp.val
    experiment.input_function.vn = slider_vn.val
    experiment.input_function.frequency = slider_frequency.val

    x_solve_ivp = solve_ivp(dxdt, (time[0], time[-1]), [x0], method="LSODA", t_eval=t, args=memristor_args)
    x = x_solve_ivp.y[0, :]

    i = I(t, x)
    v = V(t)

    # update voltage
    axes[1].set_ylim([np.min(v) - np.abs(0.5 * np.min(v)), np.max(v) + np.abs(0.5 * np.max(v))])
    lines[1].set_ydata(v)

    # update memristor
    axes[0].set_ylim([np.min(i) - np.abs(0.5 * np.min(i)), np.max(i) + np.abs(0.5 * np.max(i))])
    axes[2].set_ylim([np.min(i) - np.abs(0.5 * np.min(i)), np.max(i) + np.abs(0.5 * np.max(i))])
    lines[0].set_ydata(i)
    lines[2].set_ydata(i)
    lines[2].set_xdata(v)

    fig.canvas.draw_idle()


def reset(event):
    for sv, sm in zip(voltage_sliders, memristor_sliders):
        sv.reset()
        sm.reset()


def update_voltage(val):
    experiment.input_function.vp = slider_vp.val
    experiment.input_function.vn = slider_vn.val
    experiment.input_function.frequency = slider_frequency.val

    # update voltage
    v = V(t)
    axes[1].set_ylim([np.min(v) - np.abs(0.5 * np.min(v)), np.max(v) + np.abs(0.5 * np.max(v))])
    lines[1].set_ydata(v)
    axes[2].set_xlim([np.min(v) - np.abs(0.5 * np.min(v)), np.max(v) + np.abs(0.5 * np.max(v))])
    lines[2].set_xdata(v)

    update_memristor(0)


# TODO finish implementing time change
def update_experiment(val):
    experiment.simulation["time"] = experiment.set_time(slider_time.val)


def update_memristor(val):
    memristor_args = [sl.val for sl in memristor_sliders]

    x_solve_ivp = solve_ivp(dxdt, (time[0], time[-1]), [x0], method="LSODA", t_eval=t, args=memristor_args)
    x = x_solve_ivp.y[0, :]
    i = I(t, x)

    i_oom = order_of_magnitude.symbol(np.max(i))
    i_scaled = i * 1 / i_oom[0]
    axes[0].set_ylim([np.min(i_scaled) - np.abs(0.5 * np.min(i_scaled)),
                      np.max(i_scaled) + np.abs(0.5 * np.max(i_scaled))])
    axes[0].set_ylabel(f"Current ({i_oom[1]}A)", color="b")
    lines[0].set_ydata(i_scaled)
    axes[2].set_ylim([np.min(i_scaled) - np.abs(0.5 * np.min(i_scaled)),
                      np.max(i_scaled) + np.abs(0.5 * np.max(i_scaled))])
    lines[2].set_ydata(i_scaled)

    fig.canvas.draw_idle()


################################################
#          Event handlers registration
################################################

button_input.on_clicked(switch_input)
button_reset.on_clicked(reset)

for s in voltage_sliders:
    s.on_changed(update_voltage)
for s in experiment_sliders:
    s.on_changed(update_experiment)
for s in memristor_sliders:
    s.on_changed(update_memristor)

plt.show()
