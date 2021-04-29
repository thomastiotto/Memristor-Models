import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import ScalarFormatter

from functions import *
from models import *
from experiments import *

# TODO vary input voltage
# TODO switch input voltage type
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

fig, lines, axes = plot_memristor(v, i, t, experiment.name, figsize=(10, 6), iv_arrows=False)

# create the sliders
fig.subplots_adjust(bottom=0.3)
colour = "lightgoldenrodyellow"
sliders = []
ax_d = plt.axes([0.05, 0.15, 0.25, 0.03], facecolor=colour)
d0 = experiment.memristor.D
sd = Slider(
        ax_d,
        r"$D$",
        valmin=1e-9,
        valmax=100e-9,
        valinit=d0,
        valfmt=r"%.2E $m$"
        )
sliders.append(sd)
ax_ron = plt.axes([0.05, 0.1, 0.25, 0.03], facecolor=colour)
ron0 = experiment.memristor.RON
sron = Slider(
        ax_ron,
        r"$R_{ON}$",
        valmin=1e3,
        valmax=100e3,
        valinit=ron0,
        valfmt=r"%.2E $\Omega$"
        )
sliders.append(sron)
ax_roff = plt.axes([0.5, 0.15, 0.25, 0.03], facecolor=colour)
roff0 = experiment.memristor.ROFF
sroff = Slider(
        ax_roff,
        r"$R_{OFF}$",
        valmin=10e3,
        valmax=1000e3,
        valinit=roff0,
        valfmt=r"%.2E $\Omega$"
        )
sliders.append(sroff)
ax_mud = plt.axes([0.5, 0.1, 0.25, 0.03], facecolor=colour)
mud0 = experiment.memristor.muD
smud = Slider(
        ax_mud,
        r"$\mu_D$",
        valmin=1e-15,
        valmax=10e-14,
        valinit=mud0,
        valfmt=r"%.2E $m^2 s^{-1} V^{-1}$"
        )
sliders.append(smud)


def update(val):
    args = [sl.val for sl in sliders]

    x_solve_ivp = solve_ivp(dxdt, (time[0], time[-1]), [x0], method="LSODA", t_eval=t, args=args)
    x = x_solve_ivp.y[0, :]
    i = I(t, x)

    axes[0].set_ylim([np.min(i) - np.abs(0.5 * np.min(i)), np.max(i) + np.abs(0.5 * np.max(i))])
    axes[2].set_ylim([np.min(i) - np.abs(0.5 * np.min(i)), np.max(i) + np.abs(0.5 * np.max(i))])
    lines[0].set_ydata(i)
    lines[2].set_ydata(i)

    fig.canvas.draw_idle()


for s in sliders:
    s.on_changed(update)

ax_reset = plt.axes([0.35, 0.025, 0.1, 0.04])
button = Button(ax_reset, 'Reset', color=colour, hovercolor='0.975')


def reset(event):
    for s in sliders:
        s.reset()


button.on_clicked(reset)

plt.show()
