import random

import json
from functions import *

model = json.load(open('../../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an'))
iterations = 200

R0 = 1e8
x0 = 0.6251069761800688
setV = 3.86621037038006
resetV = -8.135891404816215
readV = -1


# random.seed(8)


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


for train_length in [1, 10, 100]:
    x_p = x0
    x_n = x0
    R_p = []
    R_n = []
    for j in tqdm(range(int(iterations / train_length))):
        if random.random() < .5:
            for _ in range(train_length):
                x_p, r_p = one_step_yakopcic(setV, x_p, readV, **model)
                x_n, r_n = one_step_yakopcic(resetV, x_n, readV, **model)
                R_p.append(r_p)
                R_n.append(r_n)
        else:
            for _ in range(train_length):
                x_p, r_p = one_step_yakopcic(resetV, x_p, readV, **model)
                x_n, r_n = one_step_yakopcic(setV, x_n, readV, **model)
                R_p.append(r_p)
                R_n.append(r_n)
    Wcombined = [x - y for x, y in zip([1 / x for x in R_p], [1 / x for x in R_n])]
    plt.plot(Wcombined, alpha=.5, label=f'{train_length} train')

plt.ylabel("w")
plt.xlabel("Pulse #")
plt.title('Differential synaptic weight')
plt.legend()
plt.show()
