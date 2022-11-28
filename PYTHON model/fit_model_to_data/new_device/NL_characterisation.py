import json
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from yakopcic_functions import *
from fit_model_to_data_functions import *

model = json.load(open('../../../fitted/fitting_pulses/new_device/regress_negative_then_positive'))


def one_step_yakopcic(voltage, x, readV):
    x = x + dxdt(voltage, x, model['Ap'], model['An'], model['Vp'], model['Vn'], model['xp'],
                 model['xn'], model['alphap'], model['alphan'], 1) * model['dt']
    if x < 0:
        x = 0
    if x > 1:
        x = 1

    i = current(readV, x,
                model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'],
                model['gmin_p'], model['bmin_p'], model['gmin_n'], model['bmin_n'])
    g = i / readV

    return x, g


def iterate_yakopcic(resetV, setV, iterations=100, plot_output=False, print_output=False):
    print_cond = lambda *a, **k: None
    if print_output:
        print_cond = print

    x0 = 0.999993124834466
    readV = -0.5

    antisetV = setV * -1 / 2
    antiresetV = resetV * -1 / 2
    antisetV = 0
    antiresetV = 0

    # random.seed(0)

    G = [current(readV, x0,
                 model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'],
                 model['gmin_p'], model['bmin_p'], model['gmin_n'], model['bmin_n']) / readV]
    X = [x0]

    for _ in tqdm(range(iterations), disable=True):
        x, g = one_step_yakopcic(resetV, X[-1], readV)
        x, g = one_step_yakopcic(antiresetV, x, readV)
        G.append(g)
        X.append(x)
    print_cond('RESET:')
    print_cond('Start resistance:', 1 / G[iterations], 'End resistance:', 1 / G[-1])
    print_cond('Start conductance:', G[0], 'End conductance:', G[-1])

    for _ in tqdm(range(iterations + 1), disable=True):
        x, g = one_step_yakopcic(setV, X[-1], readV)
        x, g = one_step_yakopcic(antisetV, x, readV)
        G.append(g)
        X.append(x)
    print_cond('SET:')
    print_cond('Start resistance:', 1 / G[iterations], 'End resistance:', 1 / G[-1])
    print_cond('Start conductance:', G[iterations], 'End conductance:', G[-1])

    print_cond('Dynamic range:', np.max(np.abs(G[:iterations + 1] - np.flip(G[iterations + 1:]))), 'S')

    G_norm = (G - np.min(G)) / (np.max(G) - np.min(G))
    NL = np.max(np.abs(G_norm[:iterations + 1] - np.flip(G_norm[iterations + 1:])))
    NL_point = np.argmax(np.abs(G_norm[:iterations + 1] - np.flip(G_norm[iterations + 1:])))
    print_cond('NL:', NL)

    if plot_output:
        fig, ax = plt.subplots()
        ax.plot(G_norm[:iterations], label='RESET', color='blue')
        ax.plot(np.flip(G_norm[iterations:]), label='SET', color='red')
        ax.plot([NL_point, NL_point], [G_norm[:iterations + 1][NL_point], np.flip(G_norm[iterations + 1:])[NL_point]],
                marker='o', color='black', linestyle='dashed')
        ax.legend()
        fig.suptitle(f'SET {setV} V, RESET {resetV} V')
        fig.show()

        R = 1 / np.array(G)
        fig, ax = plt.subplots()
        ax.plot(R[:iterations], label='RESET', color='blue')
        ax.plot(np.flip(R[iterations:]), label='SET', color='red')
        fig.show()

    return NL, NL_point, G_norm, G, X


def residuals_voltages(x, iterations):
    resetV = x[0]
    setV = x[1]

    NL, _, _, _, _ = iterate_yakopcic(resetV, setV, iterations=iterations)

    return NL


# -- initial conditions
n_iter = 100
resetV = -6.342839121380956
setV = 4.8838587394343485
readV = -0.5
initialV = setV
num_reset_pulses = num_set_pulses = n_iter
read_time = 0.1
nengo_time = 0.001
nengo_program_time = nengo_time * 0.7
nengo_read_time = nengo_time * 0.3
initial_time = 60

# simulate model and generate waveforms with higher precision if any time parameter is below the current dt
times = [read_time, nengo_time, nengo_program_time, nengo_read_time]
if any(t < model['dt'] for t in times):
    model['dt'] = min(times)

time, voltage, i, r, x = model_sim_with_params(pulse_length=nengo_program_time,
                                               resetV=resetV, numreset=num_reset_pulses,
                                               setV=setV, numset=num_set_pulses,
                                               readV=readV, read_length=nengo_read_time,
                                               init_set_length=initial_time, init_setV=initialV,
                                               **model)
peaks = find_peaks(r, voltage, readV, initial_time, dt=model['dt'], debug=False)

fig_plot = plot_images(time, voltage, i, r, x, f'Model', readV, unit='Conductance')
fig_plot.show()

G = 1 / peaks
G_norm = (G - np.min(G)) / (np.max(G) - np.min(G))
NL = np.max(np.abs(G_norm[:n_iter] - np.flip(G_norm[n_iter:])))
NL_point = np.argmax(np.abs(G_norm[:n_iter] - np.flip(G_norm[n_iter:])))
print('NL:', NL)
print('Dynamic range:', np.max(np.abs(G[:n_iter] - np.flip(G[n_iter:]))), 'S')

fig, ax = plt.subplots()
ax.plot(G_norm[:n_iter], label='RESET', color='blue')
ax.plot(np.flip(G_norm[n_iter:]), label='SET', color='red')
ax.plot([NL_point, NL_point], [G_norm[:n_iter][NL_point], np.flip(G_norm[n_iter:])[NL_point]],
        marker='o', color='black', linestyle='dashed')
ax.legend()
fig.suptitle(f'SET {setV} V, RESET {resetV} V')
fig.show()

# print(f'\nDEFAULT VOLTAGES ({setV} V, {resetV}) V:')
# iterate_yakopcic(resetV, setV, iterations=n_iter, plot_output=True, print_output=True)

# # -- target probabilities at 1 and find voltages that give that outcome
# find_voltages = optimize.least_squares(residuals_voltages, [resetV, setV],
#                                        args=[n_iter],
#                                        bounds=([-10, 0], [0, 10]),
#                                        method='dogbox', verbose=0)
# print('\nVOLTAGES GIVEN TARGET PROBABILITIES AT 1:')
# iterate_yakopcic(find_voltages.x[0], find_voltages.x[1], iterations=n_iter, plot_output=True, print_output=True)

# print('\nREDUCED VOLTAGES:')
# iterate_yakopcic(resetV / 5, setV, iterations=n_iter, plot_output=True, print_output=True)
