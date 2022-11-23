from scipy import optimize
import pprint
import copy
import json
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import functions

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
read_length = 0.1


def residuals_model_electron(x, peaks_gt, pulse_length, readV, read_length, model):
    model_upd = copy.deepcopy(model)
    model_upd['An'] = x[0]
    model_upd['xn'] = x[1]
    model_upd['alphan'] = x[2]
    model_upd['Ap'] = x[3]
    model_upd['xp'] = x[4]
    model_upd['alphap'] = x[5]

    # print('An:', x[0], 'xn:', x[1])

    time, voltage, i, r, x = functions.model_sim_with_params(pulse_length, -2, 100, 1, 100, readV, read_length, 60, 1,
                                                             progress_bar=False,
                                                             **model_upd)
    peaks_model = functions.find_peaks(r, voltage, readV, 60)

    return peaks_gt - peaks_model


# -- REGRESS MODEL PARAMETERS
bounds = (0, [np.inf, 1, np.inf, np.inf, 1, np.inf])
x0 = [model['An'] * 5, model['xn'] * 2.5, model['alphan'] * 5, model['Ap'], model['xp'], model['alphap']]
res_minimisation_electron = optimize.least_squares(residuals_model_electron, x0,
                                                   args=[data, 0.1, readV, read_length, model],
                                                   bounds=bounds,
                                                   method='trf', verbose=2)
print(f'Negative regression result:\n'
      f'An: {res_minimisation_electron.x[0]}\n'
      f'xn: {res_minimisation_electron.x[1]}\n'
      f'alphan: {res_minimisation_electron.x[2]}\n'
      f'Ap: {res_minimisation_electron.x[3]}\n'
      f'xp: {res_minimisation_electron.x[4]}\n'
      f'alphap: {res_minimisation_electron.x[5]}\n'
      )
model['An'] = res_minimisation_electron.x[0]
model['xn'] = res_minimisation_electron.x[1]
model['alphan'] = res_minimisation_electron.x[2]
model['Ap'] = res_minimisation_electron.x[3]
model['xp'] = res_minimisation_electron.x[4]
model['alphap'] = res_minimisation_electron.x[5]

# -- PLOT REGRESSED MODEL
time, voltage, i, r, x = functions.model_sim_with_params(0.1, -2, 100, 1, 100, readV, read_length, 60, 1, **model)
fig_plot_fit_electron, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(data, 'o', label='Data')
fig_plot_fit_electron = functions.plot_images(time, voltage, i, r, x, f'Model', readV,
                                              fig_plot_fit_electron)
fig_plot_fit_electron.show()
fig_plot_fit_electron_debug = functions.plot_images(time, voltage, i, r, x, f'Model', readV,
                                                    fig_plot_fit_electron,
                                                    plot_type='debug', model=model, show_peaks=True)
fig_plot_fit_electron_debug.show()

peaks_model = functions.find_peaks(r, voltage, readV, 60)
print('Average error:', np.mean(data - peaks_model))
pprint.pprint(model)
