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

experiment = oblea_sine()

time = experiment.simulation["time"]
dt = experiment.simulation["dt"]
x0 = experiment.simulation["x0"]
dxdt = experiment.functions["dxdt"]
V = experiment.functions["V"]
I = experiment.functions["I"]

## Initial values

x_solve_ivp = solve_ivp(dxdt, (time[0], time[-1]), [x0], method="LSODA", t_eval=time)

t = x_solve_ivp.t
x = x_solve_ivp.y[0, :]

v = V(t)
i = I(t, x)

fig, lines, axes = plot_memristor(v, i, t, "Yakopcic", figsize=(12, 6), iv_arrows=False)

################################################
#                       GUI
################################################

colour = "lightgoldenrodyellow"

ax_switchV = plt.axes([0.005, 0.22, 0.1, 0.04])
button_input = Button(ax_switchV, "Switch V", color=colour, hovercolor='0.975')

ax_reset = plt.axes([0.7, 0.85, 0.1, 0.04])
button_reset = Button(ax_reset, 'Reset', color=colour, hovercolor='0.975')

# create the voltage sliders
fig.subplots_adjust(left=0.15)
voltage_sliders = []
ax_vp = plt.axes([0.02, 0.3, 0.02, 0.25], facecolor=colour)
slider_vp = Slider(
        ax_vp,
        r"$V^+$",
        valmin=0,
        valmax=4,
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
        valinit=experiment.simulation["t_max"],
        valfmt=r"%.2E $s$"
        )
experiment_sliders.append(slider_time)

# create the memristor sliders
fig.subplots_adjust(bottom=0.3)
memristor_sliders = []

## I parameters
ax_a1 = plt.axes([0.05, 0.15, 0.15, 0.03], facecolor=colour)
slider_a1 = Slider(
        ax_a1,
        r"$a_1$",
        valmin=0,
        valmax=2,
        valinit=experiment.memristor.a1,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_a1)
ax_a2 = plt.axes([0.05, 0.1, 0.15, 0.03], facecolor=colour)
slider_a2 = Slider(
        ax_a2,
        r"$a_2$",
        valmin=0,
        valmax=2,
        valinit=experiment.memristor.a2,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_a2)
ax_b = plt.axes([0.05, 0.05, 0.15, 0.03], facecolor=colour)
slider_b = Slider(
        ax_b,
        r"$b$",
        valmin=0,
        valmax=1,
        valinit=experiment.memristor.b,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_b)

## g parameters
ax_Ap = plt.axes([0.3, 0.15, 0.15, 0.03], facecolor=colour)
slider_Ap = Slider(
        ax_Ap,
        r"$A_p$",
        valmin=0,
        valmax=1e10,
        valinit=experiment.memristor.Ap,
        valfmt=r"%.2E"
        )
memristor_sliders.append(slider_Ap)
ax_An = plt.axes([0.3, 0.1, 0.15, 0.03], facecolor=colour)
slider_An = Slider(
        ax_An,
        r"$A_n$",
        valmin=0,
        valmax=1e10,
        valinit=experiment.memristor.An,
        valfmt=r"%.2E"
        )
memristor_sliders.append(slider_An)
ax_Vp = plt.axes([0.3, 0.05, 0.15, 0.03], facecolor=colour)
slider_Vp = Slider(
        ax_Vp,
        r"$V_p$",
        valmin=0,
        valmax=4,
        valinit=experiment.memristor.Vp,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_Vp)
ax_Vn = plt.axes([0.3, 0.0, 0.15, 0.03], facecolor=colour)
slider_Vn = Slider(
        ax_Vn,
        r"$V_n$",
        valmin=0,
        valmax=4,
        valinit=experiment.memristor.Vn,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_Vn)

## f parameters
ax_alphap = plt.axes([0.55, 0.15, 0.15, 0.03], facecolor=colour)
slider_alphap = Slider(
        ax_alphap,
        r"$\alpha_p$",
        valmin=0,
        valmax=30,
        valinit=experiment.memristor.alphap,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_alphap)
ax_alphan = plt.axes([0.55, 0.1, 0.15, 0.03], facecolor=colour)
slider_alphan = Slider(
        ax_alphan,
        r"$\alpha_n$",
        valmin=0,
        valmax=30,
        valinit=experiment.memristor.alphan,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_alphan)
ax_xp = plt.axes([0.55, 0.05, 0.15, 0.03], facecolor=colour)
slider_xp = Slider(
        ax_xp,
        r"$x_p$",
        valmin=0,
        valmax=1,
        valinit=experiment.memristor.xp,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_xp)
ax_xn = plt.axes([0.55, 0.0, 0.15, 0.03], facecolor=colour)
slider_xn = Slider(
        ax_xn,
        r"$x_n$",
        valmin=0,
        valmax=1,
        valinit=experiment.memristor.xn,
        valfmt=r"%.2f"
        )
memristor_sliders.append(slider_xn)

## Other parameters
ax_eta = plt.axes([0.8, 0.15, 0.15, 0.03], facecolor=colour)
slider_eta = Slider(
        ax_eta,
        r"$\eta$",
        valmin=-1,
        valmax=1,
        valinit=experiment.memristor.eta,
        valstep=[-1, 1],
        valfmt=r"%.0f"
        )
memristor_sliders.append(slider_eta)


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


def update_experiment(val):
    experiment.set_time(slider_time.val)

    axes[0].set_xlim([0, slider_time.val])
    axes[1].set_xlim([0, slider_time.val])

    update_memristor(0)


def update_memristor(val):
    memristor_args = [sl.val for sl in memristor_sliders]
    time = experiment.simulation["time"]
    x_solve_ivp = solve_ivp(dxdt, (time[0], time[-1]), [x0], method="LSODA", t_eval=time, args=memristor_args)
    x = x_solve_ivp.y[0, :]
    i = I(time, x)
    v = V(time)

    i_oom = order_of_magnitude.symbol(np.max(i))
    i_scaled = i * 1 / i_oom[0]

    # remove old lines
    axes[0].lines.pop(0)
    axes[1].lines.pop(0)
    axes[2].lines.pop(0)

    axes[0].plot(experiment.simulation["time"], i_scaled, color="b")
    axes[1].plot(experiment.simulation["time"], v, color="r")

    axes[0].set_ylim([np.min(i_scaled) - np.abs(0.5 * np.min(i_scaled)),
                      np.max(i_scaled) + np.abs(0.5 * np.max(i_scaled))])
    axes[0].set_ylabel(f"Current ({i_oom[1]}A)", color="b")

    axes[2].plot(v, i_scaled, color="b")
    axes[2].set_ylim([np.min(i_scaled) - np.abs(0.5 * np.min(i_scaled)),
                      np.max(i_scaled) + np.abs(0.5 * np.max(i_scaled))])

    fig.canvas.draw()


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
