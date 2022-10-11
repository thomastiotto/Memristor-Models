import scipy.stats as stats
import random

import json
from functions import *

import numpy as np
import matplotlib.pyplot as plt


class GaussianPlot:
    def __init__(self, title="Gaussian Distribution", x_label="x", y_label=None,
                 y_limit=None, lower_bound=None, upper_bound=None,
                 with_grid=True, fill_below=True, legend_location="best"):

        self.title = title
        self.x_label = x_label
        self.y_label = y_label or "N({}|μ,σ)".format(x_label)
        self.y_limit = y_limit
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.with_grid = with_grid
        self.fill_below = fill_below
        self.legend_location = legend_location

        self.plots = []

    def plot(self, mean, std, resolution=None, legend_label=None):
        self.plots.append({
            "mean": mean,
            "std": std,
            "resolution": resolution,
            "legend_label": legend_label
        })

        return self

    def show(self):
        self._prepare_figure()
        self._draw_plots()

        plt.legend(loc=self.legend_location)
        plt.show()

    def _prepare_figure(self):
        plt.figure()
        plt.title(self.title)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        if self.y_limit is not None:
            plt.ylim(0, self.y_limit)

        if self.with_grid:
            plt.grid()

    def _draw_plots(self):
        lower_bound = self.lower_bound if self.lower_bound is not None else self._compute_lower_bound()
        upper_bound = self.upper_bound if self.upper_bound is not None else self._compute_upper_bound()

        for plot_data in self.plots:
            mean = plot_data["mean"]
            std = plot_data["std"]
            resolution = plot_data["resolution"]
            legend_label = plot_data["legend_label"]

            self._draw_plot(lower_bound, upper_bound, mean, std, resolution, legend_label)

    def _draw_plot(self, lower_bound, upper_bound, mean, std, resolution, legend_label):
        resolution = resolution or max(100, int(upper_bound - lower_bound) * 10)
        legend_label = legend_label or "μ={}, σ={}".format(mean, std)

        X = np.linspace(lower_bound, upper_bound, resolution)
        dist_X = self._distribution(X, mean, std)

        if self.fill_below: plt.fill_between(X, dist_X, alpha=0.1)
        plt.plot(X, dist_X, label=legend_label)

    def _compute_lower_bound(self):
        return np.min([plot["mean"] - 4 * plot["std"] for plot in self.plots])

    def _compute_upper_bound(self):
        return np.max([plot["mean"] + 4 * plot["std"] for plot in self.plots])

    def _distribution(self, X, mean, std):
        return 1. / (np.sqrt(2. * np.pi) * std) * np.exp(-np.power((X - mean) / std, 2.) / 2)


def one_step_yakopcic(voltage, x, readV, **params):
    x = x + dxdt(voltage, x, model['Ap'], model['An'], model['Vp'], model['Vn'], model['xp'],
                 model['xn'], model['alphap'], model['alphan'], 1) * model['dt']
    if x < 0:
        x = 0
    if x > 1:
        x = 1

    i = current(readV, x,
                params['gmax_p'], params['bmax_p'], params['gmax_n'], params['bmax_n'],
                params['gmin_p'], params['bmin_p'], params['gmin_n'], params['bmin_n'])
    r = voltage / i

    return x, r


model = json.load(open('../../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an'))
iterations = 5000

R0 = 1e8
x0 = 0.6251069761800688
setV = 3.86621037038006
resetV = -8.135891404816215
readV = -1

# random.seed(8)


fig_trains, ax_trains = plt.subplots(1, 1, figsize=(10, 10))
gaussian_plot = GaussianPlot()

# (SET,RESET)/(neg. error,pos. error) pulse lengths obtained from simulating mPES.py
for train_length in [(1, 1), (round(4.7370441230259654), round(4.691043103305757)),
                     (round(10.516604869146395), round(10.514494153790514)), (100, 100)]:
    print(train_length, ':')

    x_p = x0
    x_n = x0
    R_p = []
    R_n = []
    for j in tqdm(range(int(iterations / (train_length[0] + train_length[1]) * 2))):
        # Negative local error -> SET pulse excitatory, RESET pulse inhibitory -> Weight goes up
        if random.random() < .5:
            for _ in range(train_length[0]):
                x_p, r_p = one_step_yakopcic(setV, x_p, readV, **model)
                x_n, r_n = one_step_yakopcic(resetV, x_n, readV, **model)
                R_p.append(r_p)
                R_n.append(r_n)
        # Positive local error -> SET pulse inhibitory, RESET pulse excitatory -> Weight goes down
        else:
            for _ in range(train_length[1]):
                x_p, r_p = one_step_yakopcic(resetV, x_p, readV, **model)
                x_n, r_n = one_step_yakopcic(setV, x_n, readV, **model)
                R_p.append(r_p)
                R_n.append(r_n)
    Wcombined = [x - y for x, y in zip([1 / x for x in R_p], [1 / x for x in R_n])]

    mu = np.mean(Wcombined)
    sigma = np.std(Wcombined)
    print('Mean of combined weight: ', mu)
    print('Std of combined weight: ', sigma)
    gaussian_plot.plot(mu, sigma)
    ax_trains.plot(Wcombined, alpha=.5, label=f'{train_length} train')
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

ax_trains.set_ylabel("w")
ax_trains.set_xlabel("Pulse #")
ax_trains.legend()
fig_trains.suptitle('Differential synaptic weight')
fig_trains.show()

gaussian_plot.show()
