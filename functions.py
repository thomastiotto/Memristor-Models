import numpy as np
import scipy.signal
from order_of_magnitude import order_of_magnitude

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from block_timer.timer import Timer
from progressbar import progressbar


def euler_solver(f, t, dt, iv, I=None):
    with Timer(title="Euler"):
        print("Running Euler")
        x_sol = [iv]
        if I:
            current = [0.0]

        for t in progressbar(t[:-1]):
            if I:
                current.append(I(t, x_sol[-1]))

            x_sol.append(x_sol[-1] + f(t, x_sol[-1]) * dt)

        return (x_sol, I) if I else x_sol


def rk4_solver(f, t, dt, iv, I=None):
    with Timer(title="Runge-Kutta RK4"):
        print("Running Runge-Kutta RK4")
        x_sol = [iv]
        if I:
            current = [0.0]

        for t in progressbar(t[:-1]):
            if I:
                current.append(I(t, x_sol[-1]))

            k1 = f(t, x_sol[-1])
            k2 = f(t + dt / 2, x_sol[-1] + dt * k1 / 2)
            k3 = f(t + dt / 2, x_sol[-1] + dt * k2 / 2)
            k4 = f(t + dt, x_sol[-1] + dt * k3)

            x_sol.append(x_sol[-1] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

        return (x_sol, I) if I else x_sol


def __animate_memristor(v, i, t, fig, axes, filename):
    ax11 = axes[0]
    ax12 = axes[1]
    ax2 = axes[2]

    x11data, y11data = [], []
    x12data, y12data = [], []
    x2data, y2data = [], []

    line11, = ax11.plot([], [], color="b", animated=True)
    line12, = ax12.plot([], [], color="r", animated=True)
    line2, = ax2.plot([], [], animated=True)

    def update(frame):
        x11data.append(t[frame])
        y11data.append(i[frame])
        line11.set_data(x11data, y11data)
        x12data.append(t[frame])
        y12data.append(v[frame])
        line12.set_data(x12data, y12data)
        x2data.append(v[frame])
        y2data.append(i[frame])
        line2.set_data(x2data, y2data)

        return line11, line12, line2

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, len(t), 10), blit=True)
    ani.save(f"{filename}.mp4", writer=writer)


def __plot_memristor(v, i, t, axes):
    ax11 = axes[0]
    ax12 = axes[1]
    ax2 = axes[2]

    ax11.plot(t, i, color="b")
    ax12.plot(t, v, color="r")
    ax2.plot(v, i)

    arrows_every = len(v) // 200 if len(v) > 200 else 1
    x1 = np.ma.masked_array(v[:-1:arrows_every], (np.diff(v) > 0)[::arrows_every])
    x2 = np.ma.masked_array(v[:-1:arrows_every], (np.diff(v) < 0)[::arrows_every])
    ax2.plot(x1, i[:-1:arrows_every], 'b<')
    ax2.plot(x2, i[:-1:arrows_every], 'r>')


def plot_memristor(v, i, t, title, animated=False, filename=None):
    oom_i = order_of_magnitude.power_of_ten(np.max(i))
    i *= oom_i
    oom_t = order_of_magnitude.power_of_ten(np.max(t))
    t *= oom_t

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax11 = axes[0]
    ax11.set_ylabel(f"Current ({order_of_magnitude.symbol(oom_i, omit_x=True)}A)", color="b")
    ax11.tick_params('y', colors='b')
    ax11.set_xlim(np.min(t), np.max(t))
    ax11.set_ylim(np.min(i), np.max(i))
    ax12 = ax11.twinx()
    ax11.set_xlabel(f"Time ({order_of_magnitude.symbol(oom_t, omit_x=True)}s)")
    ax12.set_ylabel('Voltage (V)', color='r')
    ax12.tick_params('y', colors='r')
    ax12.set_xlim(np.min(t), np.max(t))
    ax12.set_ylim([np.min(v) - np.abs(0.5 * np.min(v)), np.max(v) + np.abs(0.5 * np.max(v))])
    ax2 = axes[1]
    ax2.set_xlim(np.min(v), np.max(v))
    ax2.set_ylim(np.min(i), np.max(i))
    ax2.set_ylabel(f"Current ({order_of_magnitude.symbol(oom_i, omit_x=True)}A)")
    ax2.set_xlabel("Voltage (V)")
    fig.suptitle(f"Memristor Voltage and Current vs. Time ({title})")
    fig.tight_layout()

    if animated:
        __animate_memristor(v, i, t, fig, [ax11, ax12, ax2], filename)
    else:
        __plot_memristor(v, i, t, [ax11, ax12, ax2])

    return fig


def add_arrow_to_line2D(axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>', arrowsize=1, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes:
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
            "arrowstyle"    : arrowstyle,
            "mutation_scale": 10 * arrowsize,
            }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
                arrow_tail, arrow_head, transform=transform,
                **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows


class InputVoltage():
    def __init__(self, shape, vp=1, vn=None, frequency=None, period=None, t_max=0):
        input_functions = {
                "sine"    : self.sine,
                "triangle": self.triangle
                }

        assert shape in ["sine", "triangle"]
        if shape == "triangle": assert t_max > 0
        assert frequency or period

        self.shape = shape
        self.func = input_functions[shape]
        self.vp = vp
        self.vn = vn if vn else vp
        self.frequency = 1 / period if period else frequency
        self.period = 1 / frequency if frequency else period
        self.t_max = t_max

    def sine(self, t):
        pos = self.vp * np.sin(2 * self.frequency * np.multiply(np.pi, t))
        neg = self.vn * np.sin(2 * self.frequency * np.multiply(np.pi, t))
        v = np.where(pos > 0, pos, neg)

        return v

    def triangle(self, t):

        pos = self.vp * np.abs(scipy.signal.sawtooth(2 * self.frequency * np.pi * t + np.pi / 2, 0.5))
        neg = -1 * self.vn * np.abs(scipy.signal.sawtooth(2 * self.frequency * np.pi * t + np.pi / 2, 0.5))

        if isinstance(t, np.ndarray) and len(t) > 1:
            pos[len(pos) // 2:] *= -1
        elif t > self.t_max / 2:
            pos *= -1

        v = np.where(pos > 0, pos, neg)

        return v

    def print(self, start="\t"):
        start_lv2 = start + "\t"
        print(f"{start_lv2}Shape {self.shape}")
        print(f"{start_lv2}Magnitude +{self.vp} / -{self.vn} V")
        print(f"{start_lv2}Frequency {self.frequency} Hz")
        print(f"{start_lv2}Period {self.period} s")


class WindowFunction():
    def __init__(self, type, p=1, j=1):
        window_functions = {
                "none"    : self.no_window,
                "joglekar": self.joglekar,
                "biolek"  : self.biolek,
                "anusudha": self.anusudha,
                }

        assert type in ["none", "joglekar", "biolek", "anusudha"]
        self.type = type
        self.func = window_functions[type]
        self.p = p
        self.j = j

    def no_window(self, **kwargs):
        return 1

    def joglekar(self, **kwargs):
        x = kwargs["x"]

        return 1 - np.power(np.multiply(2, x) - 1, 2 * self.p)

    def biolek(self, **kwargs):
        x = kwargs["x"]
        i = kwargs["i"]

        return 1 - np.power(x - np.heaviside(-i, 1), 2 * self.p)

    def anusudha(self, **kwargs):
        x = kwargs["x"]

        return np.multiply(self.j, 1 - np.multiply(2, np.power(np.power(x, 3) - x + 1, self.p)))

    def print(self, start="\t"):
        start_lv2 = start + "\t"
        print(f"{start_lv2}Type {self.type}")
        print(f"{start_lv2}Parameter p {self.p}")
        if self.type in ("anusudha"):
            print(f"{start_lv2}Parameter j {self.j}")
