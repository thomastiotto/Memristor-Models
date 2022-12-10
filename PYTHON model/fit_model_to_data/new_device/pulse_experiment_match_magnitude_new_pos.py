from scipy import optimize
import pprint
import copy
import json
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from fit_model_to_data_functions import *

# model tuned by hand
model = {'An': 0.2130155732,
         'Ap': 0.071,
         'Vn': 0,
         'Vp': 0,
         'alphan': 21.040384406999998,
         'alphap': 9.2,
         'bmax_n': 3.9520267779285256,
         'bmax_p': 4.988561168,
         'bmin_n': 0.02811754510744163,
         'bmin_p': 0.002125127287,
         'dt': 0.001,
         'eta': 1,
         'gmax_n': 1.7980192645432986e-06,
         'gmax_p': 0.0004338454236,
         'gmin_n': 2.630532067891402e-06,
         'gmin_p': 0.03135053798,
         'x0': 0.5196300344689345,
         'xn': 0.1433673316,
         'xp': 0.11}

# model = json.load(open('../../../fitted/fitting_pulses/new_device/regress_first_last'))
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
                                                   init_set_length=0, init_setV=0,
                                                   progress_bar=False,
                                                   **model_upd)
    peaks_model = find_peaks(r, voltage, readV, 0)

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
                                               init_set_length=0, init_setV=0,
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

peaks_model = find_peaks(r, voltage, readV, 0)
print('Average error:', np.mean(data - peaks_model))
pprint.pprint(model)
json.dump(model, open('../../../fitted/fitting_pulses/new_device/regress_positive_from_handtuned_negative', 'w'),
          indent=2)
