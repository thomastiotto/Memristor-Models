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
data = np.loadtxt('../../../raw_data/pulses/new_device/Sp1V_RSm2V_Rm500mV_processed.txt', delimiter='\t', skiprows=1,
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

# -- FIND X AFTER INITIAL SET TO SPEED UP SUCCESSIVE SIMULATIONS
time, voltage, i, r, x = model_sim_with_params(program_time, resetV, 0, setV, 0,
                                               readV, read_time,
                                               initial_time, initialV,
                                               **model)
x0 = np.max(x[int(initial_time / model['dt']):])
print("Value of x after long initial SET pulse:", x0,
      '\nUsing this value as starting point for successive simulations and skipping initial SET pulse')
model['x0'] = x0


def residuals_model_electron_neg(x, peaks_gt, pulse_length, readV, read_length, model):
    model_upd = copy.deepcopy(model)
    model_upd['gmax_n'] = x[0]
    model_upd['bmax_n'] = x[1]
    model_upd['gmin_n'] = x[2]
    model_upd['bmin_n'] = x[3]
    model_upd['gmax_p'] = x[4]
    model_upd['bmax_p'] = x[5]
    model_upd['gmin_p'] = x[6]
    model_upd['bmin_p'] = x[7]

    # print('An:', x[0], 'xn:', x[1])

    time, voltage, i, r, x = model_sim_with_params(pulse_length, resetV, num_reset_pulses, setV, num_set_pulses,
                                                   readV, read_length,
                                                   0, 0,
                                                   progress_bar=False,
                                                   **model_upd)
    peaks_model = find_peaks(r, voltage, readV, 0)

    # return np.take(peaks_gt, [0, num_reset_pulses - 1, num_reset_pulses, -1]) - np.take(peaks_model,
    #                                                                                     [0, num_reset_pulses - 1,
    #                                                                                      num_reset_pulses, -1])
    # return peaks_gt[:num_reset_pulses] - peaks_model[:num_reset_pulses]
    return peaks_gt - peaks_model


# -- REGRESS NEGATIVE MODEL PARAMETERS
bounds = (0, np.inf)
x0 = [model['gmax_n'], model['bmax_n'], model['gmin_n'], model['bmin_n'], model['gmax_p'], model['bmax_p'],
      model['gmin_p'], model['bmin_p']]
res_minimisation_electron = optimize.least_squares(residuals_model_electron_neg, x0,
                                                   args=[data, program_time, readV, read_time, model],
                                                   bounds=bounds,
                                                   # method='dogbox',
                                                   verbose=2)
print(f'Negative regression result:\n'
      f'gmax_n: {res_minimisation_electron.x[0]}\n'
      f'bmax_n: {res_minimisation_electron.x[1]}\n'
      f'gmin_n: {res_minimisation_electron.x[2]}\n'
      f'bmin_n: {res_minimisation_electron.x[3]}\n'
      f'gmax_p: {res_minimisation_electron.x[4]}\n'
      f'bmax_p: {res_minimisation_electron.x[5]}\n'
      f'gmin_p: {res_minimisation_electron.x[6]}\n'
      f'bmin_p: {res_minimisation_electron.x[7]}\n'
      )
model['gmax_n'] = res_minimisation_electron.x[0]
model['bmax_n'] = res_minimisation_electron.x[1]
model['gmin_n'] = res_minimisation_electron.x[2]
model['bmin_n'] = res_minimisation_electron.x[3]
model['gmax_p'] = res_minimisation_electron.x[4]
model['bmax_p'] = res_minimisation_electron.x[5]
model['gmin_p'] = res_minimisation_electron.x[6]
model['bmin_p'] = res_minimisation_electron.x[7]

# -- PLOT REGRESSED MODEL
time, voltage, i, r, x = model_sim_with_params(program_time, resetV, num_reset_pulses, setV, num_set_pulses,
                                               readV, read_time,
                                               0, 0,
                                               **model)
fig_plot_fit_electron, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(data, 'o', label='Data')
fig_plot_fit_electron = plot_images(time, voltage, i, r, x, f'Model', readV,
                                    fig_plot_fit_electron)
fig_plot_fit_electron.show()
fig_plot_fit_electron_debug = plot_images(time, voltage, i, r, x, f'Model', readV,
                                          fig_plot_fit_electron,
                                          plot_type='debug', model=model, show_peaks=True
                                          )
fig_plot_fit_electron_debug.show()

# -- SAVE MODEL
peaks_model = find_peaks(r, voltage, readV, 0)
print('Average error:', np.mean(data - peaks_model))
pprint.pprint(model)
# json.dump(model, open('../../../fitted/fitting_pulses/new_device/regress_first_last', 'w'), indent=2)
