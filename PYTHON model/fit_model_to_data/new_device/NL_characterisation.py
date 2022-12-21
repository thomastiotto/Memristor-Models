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
#  -- IMPORT DATA
data = np.loadtxt('../../../raw_data/pulses/new_device/Sp1V_RSm2V_Rm500mV_processed.txt', delimiter='\t', skiprows=1,
                  usecols=[2])
# -- transform data from current to resistance
data = readV / data

model = json.load(open('../../../fitted/fitting_pulses/new_device/mystery_model'))
model['dt'] = 0.0001

pprint.pprint(model)


def min_NL_max_DR(x, scaler, x0, iterations):
    NL, DR, NL_point, peaks, R, X, X_peaks, V = iterate_yakopcic(x[0], x[1],
                                                                 x0=x0,
                                                                 reset_iter=iterations, set_iter=iterations,
                                                                 plot_output=False, print_output=False)
    # -- normalise DR to [0,1]
    DR = DR / (scaler[1] - scaler[0])

    initial_point_distance = np.abs(peaks[-1] - peaks[0])
    initial_point_distance = initial_point_distance / (scaler[1] - scaler[0])

    return NL - DR


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def find_X_initial(peaks, X_peaks):
    id_initial = find_nearest(peaks, np.min(peaks) + DR / 2)
    X_initial = X_peaks[id_initial]
    R_initial = peaks[id_initial]

    print('Values nearest to centre of peaks')
    print(f'Pulse #: {id_initial}')
    print(f'X: [ {np.min(X_peaks)} | {X_initial} | {np.max(X_peaks)} ]')
    print(f'R: [ {np.min(peaks)} | {R_initial} | {np.max(peaks)} ]')

    return id_initial, X_initial, R_initial


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


def iterate_yakopcic(resetV, setV, antiresetV=None, antisetV=None, reset_iter=100, set_iter=100, x0=None,
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
    peaks, x_peaks = find_peaks(r, voltage, readV, 0, dt=model['dt'], x=x, debug=False)

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

    R_norm = (peaks - np.min(peaks)) / (np.max(peaks) - np.min(peaks))
    NL = None
    NL_point = None
    if reset_iter == set_iter:
        NL = np.max(np.abs(R_norm[:reset_iter] - np.flip(R_norm[reset_iter:])))
        NL_point = np.argmax(np.abs(R_norm[:reset_iter] - np.flip(R_norm[reset_iter:])))
        DR = np.max(peaks) - np.min(peaks)
        print_cond('NL:', NL)
        print_cond(f'Dynamic range: {DR:.2E} Ω')

    if plot_output:
        fig_plot = plot_images(time, voltage, i, r, x, f'Model', readV, unit='Resistance')
        fig_plot.show()

        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(peaks[:reset_iter], marker='o', label='RESET', color='blue')
        ax.plot(np.flip(peaks[reset_iter:]), marker='o', label='SET', color='red')
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Creating an ndarray from ragged nested sequences")
                ax.plot([NL_point, NL_point], [peaks[:reset_iter][NL_point], np.flip(peaks[reset_iter:])[NL_point]],
                        marker='o', color='black', linestyle='dashed')
        except:
            pass
        ax.legend()
        fig.suptitle(f'RESET {resetV:.2f} V SET {setV:.2f} V ')
        ax.set_title(f'NL {NL:.2f} DR {DR:.2E}', fontsize='medium')
        fig.show()

    return NL, DR, NL_point, peaks, r, x, x_peaks, voltage


resetV = -6.640464569013251
setV = 5.016534745455379

print(f'\nDEFAULT VOLTAGES {resetV} V / {setV} V')
NL, DR, NL_point, peaks, R, X, X_peaks, V = iterate_yakopcic(resetV, setV,
                                                             reset_iter=100, set_iter=100,
                                                             plot_output=True, print_output=True)

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(peaks, 'oy', label='R')
ax2.plot(X_peaks, 'og', label='X')
print('RESET')
id_initial, X_initial, R_initial = find_X_initial(peaks[:100], X_peaks[:100])
ax.plot(id_initial, R_initial, 'ok')
ax2.plot(id_initial, X_initial, 'ok', label='Initial')
print('SET')
id_initial, X_initial, R_initial = find_X_initial(peaks[100:], X_peaks[100:])
ax.plot(id_initial + 100, R_initial, 'ok')
ax2.plot(id_initial + 100, X_initial, 'ok')
fig.suptitle('Initial X and R with default voltages')
fig.legend()
fig.show()

# -- compute scaling factor
min_R = readV / current(readV, 1, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'],
                        model['bmin_p'], model['gmin_n'], model['bmin_n'])
max_R = readV / current(readV, 0, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'],
                        model['bmin_p'], model['gmin_n'], model['bmin_n'])
scaler = [min_R, max_R]

x0 = [resetV, setV]
# bounds = [(-10, -1), (0.1, 10), (0, 10), (-10, 0)]
bounds = [(-10, 1), (0, 10)]
print('MINIMISE NL AND MAXIMISE DR')
res_NL_DR = scipy.optimize.minimize(min_NL_max_DR, x0=x0, bounds=bounds, args=(scaler, None, 100))
NL, DR, NL_point, peaks, R, X, X_peaks, V = iterate_yakopcic(res_NL_DR.x[0], res_NL_DR.x[1],
                                                             reset_iter=100,
                                                             set_iter=100,
                                                             plot_output=True, print_output=True)
print('Found voltages')
print('RESET V', res_NL_DR.x[0], 'SET V', res_NL_DR.x[1])
# plt.plot(X_peaks)
# plt.show()

print('FROM CROSSOVER POINT')
cross_point = np.argmin(np.abs(peaks[:100] - np.flip(peaks[100:])))
X_cross = X_peaks[cross_point]
NL, DR, NL_point, peaks, R, X, X_peaks, V = iterate_yakopcic(res_NL_DR.x[0], res_NL_DR.x[1],
                                                             reset_iter=70, set_iter=70,
                                                             x0=X_cross * 1.25,
                                                             plot_output=True, print_output=True)

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(peaks, 'oy', label='R')
ax2.plot(X_peaks, 'og', label='X')
print('RESET')
id_initial, X_initial, R_initial = find_X_initial(peaks[:100], X_peaks[:100])
ax.plot(id_initial, R_initial, 'ok')
ax2.plot(id_initial, X_initial, 'ok', label='Initial')
print('SET')
id_initial, X_initial, R_initial = find_X_initial(peaks[100:], X_peaks[100:])
ax.plot(id_initial + 100, R_initial, 'ok')
ax2.plot(id_initial + 100, X_initial, 'ok')
fig.suptitle('Initial X and R with reduced voltages')
fig.legend()
fig.show()

res_NL_DR = scipy.optimize.minimize(min_NL_max_DR, x0=x0, bounds=bounds, args=(scaler, X_cross * 1.25, 70))
NL, DR, NL_point, peaks, R, X, X_peaks, V = iterate_yakopcic(res_NL_DR.x[0], res_NL_DR.x[1],
                                                             reset_iter=70, set_iter=70,
                                                             x0=X_cross * 1.25,
                                                             plot_output=True, print_output=True)
print('Found voltages')
print('RESET V', res_NL_DR.x[0], 'SET V', res_NL_DR.x[1])
