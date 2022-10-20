import scipy.stats as stats
import random

import json
from functions import *

import numpy as np
import matplotlib.pyplot as plt


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
    r = readV / i

    return x, r


model = json.load(open('../../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an'))
iterations = 10

x0 = 0.6251069761800688
setV = 3.86621037038006
resetV = -8.135891404816215
resetV = -0.2
readV = -0.1

# random.seed(0)


x_p = x_n = x0
R_p = []
R_n = []
X_p = [x_p]
X_n = [x_n]

for j in tqdm(range(iterations)):
    x_p, r_p = one_step_yakopcic(setV, x_p, readV, **model)
    R_p.append(r_p)
print('\nStart value:', R_p[0], 'End value:', R_p[-1])
set_efficacy = np.mean(np.diff(R_p) / np.abs(R_p[:-1]) * 100)
print('Average resistance change with SET pulses:', set_efficacy, '%')

for j in tqdm(range(iterations)):
    x_n, r_n = one_step_yakopcic(resetV, x_n, readV, **model)
    R_n.append(r_n)
print('\nStart value:', R_n[0], 'End value:', R_n[-1])
reset_efficacy = np.mean(np.diff(R_n) / np.abs(R_n[:-1]) * 100)
print('Average resistance change with RESET pulses:', reset_efficacy, '%')
print('RESET pulses are more effective than SET pulses by', ((set_efficacy - reset_efficacy) / set_efficacy) * 100, '%')

fig, ax = plt.subplots()
ax.plot(R_p, label='SET')
ax.plot(R_n, label='RESET')
ax.legend()
fig.suptitle(f'SET {setV} V, RESET {resetV} V')
fig.show()
