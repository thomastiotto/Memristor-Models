from scipy import optimize
import pprint
import copy
import json
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from fit_model_to_data_functions import *

model = {'An': 0.02662694665,
         'Ap': 0.071,
         'Vn': 0,
         'Vp': 0,
         'alphan': 0.7013461469,
         'alphap': 9.2,
         'dt': 0.001,
         'eta': 1,
         'x0': 0.0,
         'xn': 0.1433673316,
         'xp': 0.11,
         'gmax_p': 0.0004338454236,
         'bmax_p': 4.988561168,
         'gmin_p': 0.03135053798,
         'bmin_p': 0.002125127287,
         'gmax_n': 8.44e-06 / 15,
         'bmax_n': 6.272960721 / 1.1,
         'gmin_n': 1.45e-05 / 7,
         'bmin_n': 3.295533935 / 100,
         }
readV = -0.5
debug = False

pprint.pprint(model)

#  -- IMPORT DATA
data = np.loadtxt('../../../../raw_data/pulses/new_device/Sp1V_RSm2V_Rm500mV_processed.txt', delimiter='\t', skiprows=1,
                  usecols=[2])
# -- transform data from current to resistance
data = readV / data

# --  length of read pulses
# -- EXPERIMENT HYPERPARAMETNERS
resetV = -2
setV = 1
readV = -0.5
initialV = setV
num_reset_pulses = 100
num_set_pulses = 100
program_time = 0.1
initial_time = 60
read_time = 0.1


def residuals_model_electron_neg(x, peaks_gt, pulse_length, readV, read_length, model):
    model_upd = copy.deepcopy(model)
    model_upd['An'] = x[0]
    model_upd['xn'] = x[1]
    model_upd['alphan'] = x[2]

    # print('An:', x[0], 'xn:', x[1])

    time, voltage, i, r, x = model_sim_with_params(pulse_length, resetV, num_reset_pulses, setV, num_set_pulses, readV,
                                                   read_length, initial_time, initialV,
                                                   progress_bar=False,
                                                   **model_upd)
    peaks_model = find_peaks(r, voltage, readV, initial_time)

    return peaks_gt[:num_reset_pulses] - peaks_model[:num_reset_pulses]


# -- REGRESS NEGATIVE MODEL PARAMETERS
bounds = (0, [np.inf, 1, np.inf])
x0 = [model['An'] * 5, model['xn'] * 2.5, model['alphan'] * 5]
res_minimisation_electron = optimize.least_squares(residuals_model_electron_neg, x0,
                                                   args=[data, program_time, readV, read_time, model],
                                                   bounds=bounds,
                                                   method='dogbox', verbose=2)
print(f'Negative regression result:\n'
      f'An: {res_minimisation_electron.x[0]}\n'
      f'xn: {res_minimisation_electron.x[1]}\n'
      f'alphan: {res_minimisation_electron.x[2]}\n'
      )
model['An'] = res_minimisation_electron.x[0]
model['xn'] = res_minimisation_electron.x[1]
model['alphan'] = res_minimisation_electron.x[2]

# -- PLOT REGRESSED MODEL
time, voltage, i, r, x = model_sim_with_params(program_time, resetV, num_reset_pulses, setV, num_set_pulses, readV,
                                               read_time,
                                               initial_time, initialV, **model)
fig_plot_fit_electron, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(data, 'o', label='Data')
fig_plot_fit_electron = plot_images(time, voltage, i, r, x, f'Model', readV,
                                    fig_plot_fit_electron)
fig_plot_fit_electron.show()
fig_plot_fit_electron_debug = plot_images(time, voltage, i, r, x, f'Model', readV,
                                          fig_plot_fit_electron,
                                          plot_type='debug', model=model, show_peaks=True)
fig_plot_fit_electron_debug.show()

peaks_model = find_peaks(r, voltage, readV, initial_time)
print('Average error:', np.mean(data - peaks_model))
pprint.pprint(model)


def residuals_model_electron_pos(x, peaks_gt, pulse_length, readV, read_length, model):
    model_upd = copy.deepcopy(model)
    model_upd['Ap'] = x[0]
    model_upd['xp'] = x[1]
    model_upd['alphap'] = x[2]

    # print('An:', x[0], 'xn:', x[1])

    time, voltage, i, r, x = model_sim_with_params(pulse_length=pulse_length,
                                                   resetV=resetV, numreset=num_reset_pulses,
                                                   setV=setV, numset=num_set_pulses,
                                                   readV=readV, read_length=read_length,
                                                   init_set_length=initial_time, init_setV=initialV,
                                                   progress_bar=False,
                                                   **model_upd)
    peaks_model = find_peaks(r, voltage, readV, initial_time)

    return peaks_gt[num_reset_pulses:] - peaks_model[num_reset_pulses:]


# -- REGRESS POSITIVE MODEL PARAMETERS
bounds = ([0, 0, 0], [np.inf, 1, np.inf])
x0 = [model['Ap'], model['xp'], model['alphap']]
res_minimisation_electron = optimize.least_squares(residuals_model_electron_pos, x0,
                                                   args=[data, program_time, readV, read_time, model],
                                                   bounds=bounds,
                                                   method='dogbox', verbose=2)
print(f'Positive regression result:\n'
      f'Ap: {res_minimisation_electron.x[0]}\n'
      f'xp: {res_minimisation_electron.x[1]}\n'
      f'alphap: {res_minimisation_electron.x[2]}\n'
      )
model['Ap'] = res_minimisation_electron.x[0]
model['xp'] = res_minimisation_electron.x[1]
model['alphap'] = res_minimisation_electron.x[2]

# -- PLOT REGRESSED MODEL
time, voltage, i, r, x = model_sim_with_params(pulse_length=program_time,
                                               resetV=resetV, numreset=num_reset_pulses,
                                               setV=setV, numset=num_set_pulses,
                                               readV=readV, read_length=read_time,
                                               init_set_length=initial_time, init_setV=initialV,
                                               **model)
fig_plot_fit_electron, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(data, 'o', label='Data')
fig_plot_fit_electron = plot_images(time, voltage, i, r, x, f'Model', readV,
                                    fig_plot_fit_electron)
fig_plot_fit_electron.show()
fig_plot_fit_electron_debug = plot_images(time, voltage, i, r, x, f'Model', readV,
                                          fig_plot_fit_electron,
                                          plot_type='debug', model=model, show_peaks=True)
fig_plot_fit_electron_debug.show()

peaks_model = find_peaks(r, voltage, readV, initial_time)
print('Average error:', np.mean(data - peaks_model))
pprint.pprint(model)
json.dump(model, open('../../../fitted/fitting_pulses/new_device/regress_negative_then_positive', 'w'), indent=2)
