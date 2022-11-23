import random
import time
from experiment_setup import *
from yakopcic_functions import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

model = Memristor_Alina
R0 = 1e8
V_pos = 2

random.seed(8)


def compact_learning(V, x):
    X = np.zeros(V.shape, dtype=float)
    X[0] = x

    for j in range(1, len(X)):
        X[j] = X[j - 1] + dxdt(V[j], X[j - 1], model['Ap'], model['An'], model['Vp'], model['Vn'], model['xp'],
                               model['xn'], model['alphap'], model['alphan'], 1) * model['dt']
        if X[j] < 0:
            X[j] = 0
        if X[j] > 1:
            X[j] = 1

    I = current(V, X, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'],
                model['bmin_p'],
                model['gmin_n'], model['bmin_n'])
    R = np.divide(V, I, out=np.zeros(V.shape, dtype=float), where=I != 0)

    # print(f'X[-1] = {X[-1]}')
    # print(f'R[-1] = {R[-1]}')

    return X[-1], R[-1]


for V_neg in tqdm(np.linspace(-4, -4, 1), position=0, leave=True, desc='RESET Sweep', colour='green', ncols=80):
    R_pos = [R0]
    R_neg = [R0]
    X_pos = 0
    X_neg = 0

    for i in tqdm(range(1000), position=0, leave=True, desc='Pulse simulation', colour='red', ncols=80):
        v_pos = generate_wave(
            {"t_rise": .001, "t_on": .1, "t_fall": .001, "t_off": .4, "V_on": V_pos, "V_off": -.1, "n_cycles": 1},
            model['dt'], 0)[1]
        v_neg = generate_wave(
            {"t_rise": .001, "t_on": .1, "t_fall": .001, "t_off": .4, "V_on": V_neg, "V_off": -.1, "n_cycles": 1},
            model['dt'], 0)[1]

        # print('vpos:', v_pos)
        # print('vneg:', v_neg)

        if random.random() < .5:
            X_pos, r_pos = compact_learning(v_pos, X_pos)
            X_neg, r_neg = compact_learning(v_neg, X_neg)
        else:
            X_pos, r_pos = compact_learning(v_neg, X_pos)
            X_neg, r_neg = compact_learning(v_pos, X_neg)

        # print(f'xp = {X_pos}')
        # print(f'xn = {X_neg}')
        R_pos.insert(0, r_pos)
        R_neg.insert(0, r_neg)
        time.sleep(0.01)

    R_pos.reverse()
    R_neg.reverse()

    Wcombined = [x - y for x, y in zip([1 / x for x in R_pos], [1 / x for x in R_neg])]
    plt.plot(Wcombined)
    plt.ylabel("w")
    plt.xlabel("Pulse #")
    # print(f'Wcombined = {Wcombined}')

plt.show()
