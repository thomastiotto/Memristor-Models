import copy

from scipy import optimize
import json
from order_of_magnitude import order_of_magnitude
import matplotlib.pyplot as plt
import pprint

from fit_model_to_data_functions import *

# -- model found with pulse_experiment_match_magnitude_new_pos_neg.py
# model = json.load(open('../../../fitted/fitting_pulses/new_device/regress_negative_then_positive'))
# model = json.load(open('../../../fitted/fitting_pulses/new_device/regress_positive_from_handtuned_negative'))
model = json.load(open('../../../fitted/fitting_pulses/new_device/mystery_model'))

pprint.pprint(model)

# -- EXPERIMENT HYPERPARAMETNERS
resetV = -2
setV = 1
readV = -0.5
initialV = setV
num_reset_pulses = 100
num_set_pulses = 100
read_time = 0.1
program_time = 0.1
nengo_time = 0.001
nengo_program_time = nengo_time * 0.7
nengo_read_time = nengo_time * 0.3
# nengo_program_time = nengo_read_time = nengo_time
initial_time = 60

# simulate model and generate waveforms with higher precision if any time parameter is below the current dt
times = [read_time, program_time, nengo_time, nengo_program_time, nengo_read_time]
if any(t < model['dt'] for t in times):
    model['dt'] = min(times)

#  -- IMPORT DATA
data = np.loadtxt('../../../raw_data/pulses/new_device/Sp1V_RSm2V_Rm500mV_processed.txt', delimiter='\t', skiprows=1,
                  usecols=[2])
# -- transform data from current to resistance
data = readV / data

fig_plot_opt, ax_plot = plt.subplots(1, 1, figsize=(6, 5))
for i in range(0, 10):
    perturbed_model = copy.deepcopy(model)
    for key in perturbed_model:
        if key not in ['dt', 'eta']:
            perturbed_model[key] = get_truncated_normal(perturbed_model[key], perturbed_model[key] * 0.15,
                                                        0, np.inf,
                                                        1, 1)

    # -- PLOT REGRESSED MODEL
    time_opt, voltage_opt, i_opt, r_opt, x_opt = model_sim_with_params(pulse_length=nengo_program_time,
                                                                       resetV=-6.640464569013251,
                                                                       numreset=num_reset_pulses,
                                                                       setV=5.016534745455379, numset=num_set_pulses,
                                                                       readV=readV, read_length=nengo_read_time,
                                                                       init_set_length=0, init_setV=0,
                                                                       **perturbed_model)
    fig_plot_opt = plot_images(time_opt, voltage_opt, i_opt, r_opt, x_opt,
                               readV=readV,
                               fig=fig_plot_opt)
fig_plot_opt.show()

time_opt, voltage_opt, i_opt, r_opt, x_opt = model_sim_with_params(pulse_length=nengo_program_time,
                                                                   resetV=-6.640464569013251,
                                                                   numreset=num_reset_pulses,
                                                                   setV=5.016534745455379, numset=num_set_pulses,
                                                                   readV=readV, read_length=nengo_read_time,
                                                                   init_set_length=0, init_setV=0,
                                                                   **model)

fig_plot, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(data, 'o',label='data')
fig_plot = plot_images(time_opt, voltage_opt, i_opt, r_opt, x_opt,
                           readV=readV,
                       fig=fig_plot)

peaks_opt = find_peaks(r_opt, voltage_opt, readV, 0, dt=model['dt'])
# compute error between model and data
print(
    f'Average error from data: {order_of_magnitude.convert(np.round(np.mean(data - peaks_opt), 2), scale="mega")[0]} MOhm ({np.round(absolute_mean_percent_error(data, peaks_opt), 2)} %)')
fig_plot.show()