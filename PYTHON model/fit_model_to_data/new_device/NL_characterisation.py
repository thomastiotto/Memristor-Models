import json
import pprint
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from yakopcic_functions import *
from fit_model_to_data_functions import *

readV = -0.5
#  -- IMPORT DATA
data = np.loadtxt('../../../raw_data/pulses/new_device/Sp1V_RSm2V_Rm500mV_processed.txt', delimiter='\t', skiprows=1,
                  usecols=[2])
# -- transform data from current to resistance
data = readV / data

model = json.load(open('../../../fitted/fitting_pulses/new_device/mystery_model'))

pprint.pprint(model)


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


def iterate_yakopcic(resetV, setV, antiresetV=1, antisetV=1, reset_iter=100, set_iter=100, x0=None,
                     plot_output=False, print_output=False):
    print_cond = lambda *a, **k: None
    if print_output:
        print_cond = print

    # -- initial conditions
    resetV = resetV
    setV = setV
    nengo_time = 0.001
    nengo_program_time = nengo_time * 0.7
    nengo_read_time = nengo_time * 0.3
    if x0 is not None:
        model['x0'] = x0

    # simulate model and generate waveforms with higher precision if any time parameter is below the current dt
    times = [nengo_time, nengo_program_time, nengo_read_time]
    if any(t < model['dt'] for t in times):
        model['dt'] = min(times)

    time, voltage, i, r, x = model_sim_with_params(pulse_length=nengo_program_time,
                                                   resetV=resetV, numreset=reset_iter,
                                                   setV=setV, numset=set_iter,
                                                   readV=readV, read_length=nengo_read_time,
                                                   init_set_length=0, init_setV=0,
                                                   antiresetV=antiresetV, antisetV=antisetV,
                                                   progress_bar=False,
                                                   **model)
    peaks = find_peaks(r, voltage, readV, 0, dt=model['dt'], debug=False)

    print_cond('x0:', x[0])
    if reset_iter > 0:
        print_cond(
            f'RESET change: {peaks[0]} Ω → {peaks[reset_iter - 1]} Ω ({absolute_mean_percent_error(peaks[0], peaks[reset_iter - 1])} % change)')
        if set_iter > 0:
            print_cond(
                f'SET change: {peaks[reset_iter - 1]} Ω → {peaks[-1]} Ω ({absolute_mean_percent_error(peaks[reset_iter - 1], peaks[-1])} % change)')
    else:
        print_cond(
            f'SET change: {peaks[0]} Ω → {peaks[-1]} Ω ({absolute_mean_percent_error(peaks[0], peaks[-1])} % change)')

    G = 1 / peaks
    G_norm = (G - np.min(G)) / (np.max(G) - np.min(G))
    NL = None
    NL_point = None
    if reset_iter == set_iter:
        NL = np.max(np.abs(G_norm[:reset_iter] - np.flip(G_norm[reset_iter:])))
        NL_point = np.argmax(np.abs(G_norm[:reset_iter] - np.flip(G_norm[reset_iter:])))
        print_cond('NL:', NL)
        print_cond('Dynamic range:', np.max(np.abs(G[:reset_iter] - np.flip(G[reset_iter:]))), 'S')

    if plot_output:
        fig_plot = plot_images(time, voltage, i, r, x, f'Model', readV, unit='Resistance')
        fig_plot.show()

        fig, ax = plt.subplots()
        ax.plot(G_norm[:reset_iter], marker='o', label='RESET', color='blue')
        ax.plot(np.flip(G_norm[reset_iter:]), marker='o', label='SET', color='red')
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Creating an ndarray from ragged nested sequences")
                ax.plot([NL_point, NL_point], [G_norm[:reset_iter][NL_point], np.flip(G_norm[reset_iter:])[NL_point]],
                        marker='o', color='black', linestyle='dashed')
        except:
            pass
        ax.legend()
        fig.suptitle(f'SET {setV} V, RESET {resetV} V')
        fig.show()

    return NL, NL_point, peaks, 1 / G, x, voltage, G


resetV = -6.640464569013251
setV = 5.016534745455379
print(f'\nDEFAULT VOLTAGES ({resetV}) V / {setV} V')
NL, NL_point, peaks, R, X, V, G = iterate_yakopcic(resetV, setV, reset_iter=100, set_iter=1000,
                                                   plot_output=True, print_output=True)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(X[:200])
ax.twinx().plot(V[:200], 'r')
fig.show()
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(X[750:950])
ax.twinx().plot(V[750:950], 'r')
fig.show()
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(X)
fig.show()

# print('CHECK NONLINEARITY')
# for rs in np.arange(1, -0.1, -0.1):
#     iterate_yakopcic(resetV, setV, reset_iter=2, set_iter=0, x0=rs, plot_output=False, print_output=True)
# for rs in np.arange(0, 1.1, 0.1):
#     iterate_yakopcic(resetV, setV, reset_iter=0, set_iter=2, x0=rs, plot_output=False, print_output=True)

# iterate_yakopcic(resetV, setV, reset_iter=100, set_iter=0, plot_output=True, print_output=True)
# iterate_yakopcic(resetV, setV, reset_iter=0, set_iter=100, plot_output=True, print_output=True)

# iterate_yakopcic(resetV, setV, reset_iter=100, set_iter=100, x0=1, plot_output=True, print_output=True)
# iterate_yakopcic(resetV, setV, reset_iter=100, set_iter=100, x0=0, plot_output=True, print_output=True)

# # -- target probabilities at 1 and find voltages that give that outcome
# find_voltages = optimize.least_squares(residuals_voltages, [resetV, setV],
#                                        args=[n_iter],
#                                        bounds=([-10, 0], [0, 10]),
#                                        method='dogbox', verbose=0)
# print('\nVOLTAGES GIVEN TARGET PROBABILITIES AT 1:')
# iterate_yakopcic(find_voltages.x[0], find_voltages.x[1], iterations=n_iter, plot_output=True, print_output=True)

# print('\nREDUCED VOLTAGES:')
# iterate_yakopcic(resetV / 5, setV, iterations=n_iter, plot_output=True, print_output=True)
