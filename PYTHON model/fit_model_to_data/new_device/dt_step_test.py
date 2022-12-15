import json
import pprint

import scipy.optimize
import sklearn
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm
from yakopcic_functions import *
from fit_model_to_data_functions import *

readV = -0.5

model = json.load(open('../../../fitted/fitting_pulses/new_device/mystery_model'))

pprint.pprint(model)

# -- initial conditions
resetV = -6.4688295585009605
setV = 0.24694177778629942
x0 = 0.0032239748515913717

num_pulses = 7
readV = -0.5
voltage = np.array([resetV] * (num_pulses + 1))
x = np.zeros(voltage.shape, dtype=float)
x[0] = x0
dt = 0.0001

for j in tqdm(range(1, len(x))):
    x[j] = x[j - 1] + dxdt(voltage[j], x[j - 1],
                           model['Ap'], model['An'], model['Vp'], model['Vn'],
                           model['xp'], model['xn'], model['alphap'], model['alphan'], 1) * dt
i = current(readV, x,
            model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'],
            model['gmin_p'], model['bmin_p'], model['gmin_n'], model['bmin_n'])
r = np.divide(readV, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)

x_short = x
r_short = r

voltage = np.array([resetV] * 2)
x = np.zeros(voltage.shape, dtype=float)
x[0] = x0

for j in tqdm(range(1, len(x))):
    x[j] = x[j - 1] + dxdt(voltage[j], x[j - 1],
                           model['Ap'], model['An'], model['Vp'], model['Vn'],
                           model['xp'], model['xn'], model['alphap'], model['alphan'], 1) * (dt * num_pulses)
i = current(readV, x,
            model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'],
            model['gmin_p'], model['bmin_p'], model['gmin_n'], model['bmin_n'])
r = np.divide(readV, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)

x_long = x
r_long = r

min_R = readV / current(readV, 1, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'],
                        model['bmin_p'], model['gmin_n'], model['bmin_n'])
max_R = readV / current(readV, 0, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'],
                        model['bmin_p'], model['gmin_n'], model['bmin_n'])
scaler = [min_R, max_R]

r_short_norm = (r_short[-1] - r_short[0]) / (scaler[1] - scaler[0])
r_long_norm = (r_long[-1] - r_long[0]) / (scaler[1] - scaler[0])

print('Short dt final X', x_short[-1], 'giving final R', r_short[-1])
print('Long dt final X', x_long[-1], 'giving final R', r_long[-1])
print('Absolute percentage error in X', absolute_mean_percent_error(x_short[-1], x_long[-1]), '%')
print('Absolute percentage error in R', absolute_mean_percent_error(r_short[-1], r_long[-1]), '%')

plt.plot(x_short, 'o')
plt.plot([0, len(x_short) - 1], x_long, 'o')
plt.show()
