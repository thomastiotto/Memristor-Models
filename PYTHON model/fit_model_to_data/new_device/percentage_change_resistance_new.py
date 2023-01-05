import json
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from yakopcic_functions import *
from fit_model_to_data_functions import *

iterations = 10

new_model_pack = {
    'model': json.load(open('../../../fitted/fitting_pulses/new_device/mystery_model')),
    'resetV': -6.4688295585009605,
    'setV': 0.24694177778629942,
    'readV': -0.5,
    'x0': 0.003212612055682041,
    'n_iter': iterations
}
old_model_pack = {
    'model': json.load(open('../../../fitted/fitting_pulses/old_device/regress_negative_xp_alphap-adjusted_ap_an')),
    'resetV': -8.135891404816215,
    'setV': 3.86621037038006,
    'readV': -0.1,
    'x0': 0.6251069761800688,
    'n_iter': iterations
}


def one_step_yakopcic(voltage, x, model_pack):
    x = x + dxdt(voltage, x, model_pack['model']['Ap'], model_pack['model']['An'], model_pack['model']['Vp'],
                 model_pack['model']['Vn'], model_pack['model']['xp'],
                 model_pack['model']['xn'], model_pack['model']['alphap'], model_pack['model']['alphan'], 1) * \
        model_pack['model']['dt']
    if x < 0:
        x = 0
    if x > 1:
        x = 1

    i = current(model_pack['readV'], x,
                model_pack['model']['gmax_p'], model_pack['model']['bmax_p'], model_pack['model']['gmax_n'],
                model_pack['model']['bmax_n'],
                model_pack['model']['gmin_p'], model_pack['model']['bmin_p'], model_pack['model']['gmin_n'],
                model_pack['model']['bmin_n'])
    r = model_pack['readV'] / i

    return x, r


def iterate_yakopcic(resetV, setV, model_pack, iterations=None, x0=None, plot_output=False, print_output=False):
    print_cond = lambda *a, **k: None
    if print_output:
        print_cond = print

    if x0 is None:
        x0 = model_pack['x0']
    if iterations is None:
        iterations = model_pack['n_iter']

    # random.seed(0)

    x_p = x_n = x0
    R_p = [model_pack['readV'] / current(model_pack['readV'], x0,
                                         model_pack['model']['gmax_p'], model_pack['model']['bmax_p'],
                                         model_pack['model']['gmax_n'], model_pack['model']['bmax_n'],
                                         model_pack['model']['gmin_p'], model_pack['model']['bmin_p'],
                                         model_pack['model']['gmin_n'], model_pack['model']['bmin_n'])]
    R_n = [model_pack['readV'] / current(model_pack['readV'], x0,
                                         model_pack['model']['gmax_p'], model_pack['model']['bmax_p'],
                                         model_pack['model']['gmax_n'], model_pack['model']['bmax_n'],
                                         model_pack['model']['gmin_p'], model_pack['model']['bmin_p'],
                                         model_pack['model']['gmin_n'], model_pack['model']['bmin_n'])]
    X_p = [x_p]
    X_n = [x_n]

    for _ in tqdm(range(iterations), disable=not print_output):
        x_p, r_p = one_step_yakopcic(setV, x_p, model_pack)
        R_p.append(r_p)
        X_p.append(x_p)
    print_cond('SET:')
    print_cond('Start value:', R_p[0], 'End value:', R_p[-1])
    # calculate average percent change in resistances (https://sciencing.com/calculate-mean-change-5953798.html)
    set_efficacy = absolute_mean_percent_error(R_p[0], R_p[-1])
    print_cond(f'Average resistance change with SET pulses: {set_efficacy} % ({R_p[-1] - R_p[0]} Ohm)')

    for _ in tqdm(range(iterations), disable=not print_output):
        x_n, r_n = one_step_yakopcic(resetV, x_n, model_pack)
        R_n.append(r_n)
        X_n.append(x_n)
    print_cond('RESET:')
    print_cond('Start value:', R_n[0], 'End value:', R_n[-1])
    reset_efficacy = absolute_mean_percent_error(R_n[0], R_n[-1])
    print_cond(f'Average resistance change with RESET pulses: {reset_efficacy} % ({R_n[-1] - R_n[0]} Ohm)')

    np.min(current(model_pack['readV'], np.array([X_p, X_n]),
                   model_pack['model']['gmax_p'], model_pack['model']['bmax_p'],
                   model_pack['model']['gmax_n'], model_pack['model']['bmax_n'],
                   model_pack['model']['gmin_p'], model_pack['model']['bmin_p'],
                   model_pack['model']['gmin_n'], model_pack['model']['bmin_n']))

    if set_efficacy > reset_efficacy:
        reset_vs_set_efficacy_percent = absolute_mean_percent_error(reset_efficacy, set_efficacy)
        print_cond('SET pulses are more effective than RESET pulses by', reset_vs_set_efficacy_percent, '%')
    else:
        set_vs_reset_efficacy_percent = absolute_mean_percent_error(set_efficacy, reset_efficacy)

        print_cond('RESET pulses are more effective than SET pulses by', set_vs_reset_efficacy_percent, '%')

    if set_efficacy > reset_efficacy:
        k = np.abs(set_efficacy / reset_efficacy)
        Preset = 1
        Pset = 1 / k
    else:
        k = np.abs(reset_efficacy / set_efficacy)
        Pset = 1
        Preset = 1 / k

    DR = max(np.abs(R_p[-1] - R_p[0]), np.abs(R_n[-1] - R_n[0]))

    print_cond('k:', k)
    print_cond(f'DR: {DR}')
    print_cond('Vreset', resetV, '| Vset', setV)
    print_cond('Preset:', Preset, '| Pset:', Pset)

    if plot_output:
        fig, ax = plt.subplots()
        ax.plot(R_p, label='SET')
        ax.plot(R_n, label='RESET')
        ax.legend()
        fig.suptitle(f'RESET {resetV:.2f} V | SET {setV:.2f} V')
        ax.set_title(f'k={k:.2f} | DR={DR:.2f} Ohm')
        fig.tight_layout()
        fig.show()

    return k, DR, setV, resetV, Pset, Preset


# -- objective function to minimise
def residuals_voltages(x, model_pack):
    k, _, _, _, _, _ = iterate_yakopcic(x[0], x[1], model_pack)

    return k


# -- force dynamic range to be less than threshold
def constraint_DR(x, model_pack, threshold):
    _, DR, _, _, _, _ = iterate_yakopcic(x[0], x[1], model_pack)

    return -DR + threshold


# -- force resetV to be less than readV-0.5V
def constraint_V(x, model_pack):
    return - x[0] + model_pack['readV'] + 0.5


# # -- calculate probabilities given voltages
# print('\nPROBABILITIES GIVEN DEFAULT VOLTAGES:')
# iterate_yakopcic(resetV, setV, iterations=n_iter, x0=x0,
#                  plot_output=True, print_output=True)

# -- target probabilities at 1 and find voltages that give that outcome
find_voltages = optimize.minimize(residuals_voltages,
                                  x0=[-0.2, old_model_pack['setV']],
                                  args=old_model_pack,
                                  bounds=([-10, 0], [0, 10]),
                                  # constraints=(
                                  #     {'type': 'ineq',
                                  #      'fun': constraint_V,
                                  #      'args': [old_model_pack]}),
                                  )
print('\nOLD MODEL - GET DR:')
_, DR_old_model, _, _, _, _ = iterate_yakopcic(find_voltages.x[0], find_voltages.x[1], old_model_pack,
                                               plot_output=True, print_output=True)

find_voltages = optimize.minimize(residuals_voltages,
                                  # x0=[find_voltages.x[0], find_voltages.x[1]],
                                  x0=[new_model_pack['resetV'], new_model_pack['setV']],
                                  args=new_model_pack,
                                  bounds=([-10, 0], [0, 10]),
                                  constraints=(
                                      {'type': 'ineq',
                                       'fun': constraint_V,
                                       'args': [new_model_pack]}),
                                  )
print('\nNEW MODEL - REGRESS VOLTAGES:')
iterate_yakopcic(find_voltages.x[0], find_voltages.x[1], new_model_pack,
                 plot_output=True, print_output=True)

find_voltages = optimize.minimize(residuals_voltages,
                                  # x0=[find_voltages.x[0], find_voltages.x[1]],
                                  x0=[new_model_pack['resetV'], new_model_pack['setV']],
                                  args=new_model_pack,
                                  bounds=([-10, 0], [0, 10]),
                                  constraints=({'type': 'ineq',
                                                'fun': constraint_DR,
                                                'args': (new_model_pack, DR_old_model / 3)},
                                               {'type': 'ineq',
                                                'fun': constraint_V,
                                                'args': [new_model_pack]}),
                                  options={'disp': False}
                                  )
print('\nNEW MODEL - REGRESS VOLTAGES WITH LIMIT ON DR:')
iterate_yakopcic(find_voltages.x[0], find_voltages.x[1], new_model_pack,
                 plot_output=True, print_output=True)
