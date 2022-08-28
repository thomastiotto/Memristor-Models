import random

from experiment_setup import *
from yakopcic_functions import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

model = Memristor_Alina
R0 = 1e8
vp = 0.1


def compact_learning(v, x):
    x = x + dxdt(v, x, model['Ap'], model['An'], model['Vp'], model['Vn'], model['xp'],
                 model['xn'], model['alphap'], model['alphan'], 1) * model['dt']
    if x < 0:
        x = 0
    if x > 1:
        x = 1

    i = current(v, x, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'],
                model['bmin_p'],
                model['gmin_n'], model['bmin_n'])
    return x, v / i


for vn in tqdm(np.arange(-4, -1, 0.1)):
    R_p = [R0]
    R_n = [R0]
    x_p = 0
    x_n = 0
    for j in range(5000):

        if random.random() < .5:
            x_p, r_p = compact_learning(vp, x_p)
            R_p.insert(0, r_p)
            x_n, r_n = compact_learning(vn, x_n)
            R_n.insert(0, r_n)
        else:
            x_p, r_p = compact_learning(vn, x_p)
            R_p.insert(0, r_p)
            x_n, r_n = compact_learning(vp, x_n)
            R_n.insert(0, r_n)
    R_p.reverse()
    R_n.reverse()
    Wcombined = [x - y for x, y in zip([1 / x for x in R_p], [1 / x for x in R_n])]
    plt.plot(Wcombined, label=str(vn))
    plt.ylabel("w")
    plt.xlabel("Pulse #")
    #plt.legend()
plt.show()