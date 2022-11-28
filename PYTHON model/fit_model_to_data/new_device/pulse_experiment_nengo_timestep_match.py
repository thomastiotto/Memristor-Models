from scipy import optimize
import json
from order_of_magnitude import order_of_magnitude
import matplotlib.pyplot as plt

from fit_model_to_data_functions import *

# -- model found with pulse_experiment_match_magnitude_new_pos_neg.py
model = json.load(open('../../../fitted/fitting_pulses/new_device/regress_negative_then_positive'))

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


def residuals(x, peaks_gt, pulse_length, readV, read_length, model):
    resetV, setV = x

    time, voltage, i, r, x = model_sim_with_params(pulse_length=pulse_length,
                                                   resetV=resetV, numreset=num_reset_pulses,
                                                   setV=setV, numset=num_set_pulses,
                                                   readV=readV, read_length=read_length,
                                                   init_set_length=initial_time, init_setV=initialV,
                                                   progress_bar=False,
                                                   **model)
    peaks_model = find_peaks(r, voltage, readV, initial_time, dt=model['dt'])

    return peaks_gt - peaks_model


#  -- IMPORT DATA
data = np.loadtxt('../../../raw_data/pulses/new_device/Sp1V_RSm2V_Rm500mV_processed.txt', delimiter='\t', skiprows=1,
                  usecols=[2])
# -- transform data from current to resistance
data = readV / data

# -- define ground truth
print('------------------ ORIGINAL ------------------')
time_gt, voltage_gt, i_gt, r_gt, x_gt = model_sim_with_params(pulse_length=program_time,
                                                              resetV=resetV, numreset=num_reset_pulses,
                                                              setV=setV, numset=num_set_pulses,
                                                              readV=readV, read_length=read_time,
                                                              init_set_length=initial_time, init_setV=initialV,
                                                              **model)
peaks_gt = find_peaks(r_gt, voltage_gt, readV, initial_time, dt=model['dt'])
fig_plot_fit_electron, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(data, 'o', label='Data')
fig_plot_fit_electron = plot_images(time_gt, voltage_gt, i_gt, r_gt, x_gt, f'Model', readV,
                                    fig_plot_fit_electron, dt=model['dt'])
fig_plot_fit_electron.show()
fig_plot_opt_debug = plot_images(time_gt, voltage_gt, i_gt, r_gt, x_gt, f'{resetV} V / {setV} V', readV,
                                 plot_type='debug', model=model, show_peaks=True, dt=model['dt'])
fig_plot_opt_debug.show()
print(
    f'Average error: {order_of_magnitude.convert(np.round(np.mean(data - peaks_gt), 2), scale="mega")[0]} MOhm ({np.round(absolute_mean_percent_error(data, peaks_gt), 2)} %)')

# -- run optimisation
print('------------------ OPTIMISATION ------------------')
bounds = ([-20, setV], [resetV, 20])
x0 = [bounds[1][0], bounds[0][1]]
res_minimisation = optimize.least_squares(residuals, x0,
                                          args=[peaks_gt, nengo_program_time, readV, nengo_read_time, model],
                                          bounds=bounds,
                                          method='dogbox', verbose=2)
print(f'Optimisation result:\nVreset: {res_minimisation.x[0]}\nVset: {res_minimisation.x[1]}')

# -- PLOT REGRESSED MODEL
fig_plot_opt, ax_plot = plt.subplots(1, 1, figsize=(6, 5))
ax_plot.plot(data, 'o', fillstyle='none', label=f'Data (WRITE 0.1 s / READ 0.1 s)', )
time_opt, voltage_opt, i_opt, r_opt, x_opt = model_sim_with_params(pulse_length=nengo_program_time,
                                                                   resetV=res_minimisation.x[0],
                                                                   numreset=num_reset_pulses,
                                                                   setV=res_minimisation.x[1], numset=num_set_pulses,
                                                                   readV=readV, read_length=nengo_read_time,
                                                                   init_set_length=initial_time, init_setV=initialV,
                                                                   **model)
fig_plot_opt = plot_images(time_gt, voltage_gt, i_gt, r_gt, x_gt,
                           f'Model (WRITE {program_time} s / READ {read_time} s)', readV,
                           fig_plot_opt)
fig_plot_opt = plot_images(time_opt, voltage_opt, i_opt, r_opt, x_opt,
                           f'Model (WRITE {nengo_program_time} s / READ {nengo_read_time} s)',
                           readV,
                           fig_plot_opt)
fig_plot_opt.show()
fig_plot_opt_debug = plot_images(time_opt, voltage_opt, i_opt, r_opt, x_opt, f'{resetV} V / {setV} V', readV,
                                 plot_type='debug', model=model, show_peaks=True, consider_from=initial_time,
                                 dt=model['dt'])
fig_plot_opt_debug.show()
peaks_opt = find_peaks(r_opt, voltage_opt, readV, initial_time, dt=model['dt'])
print(
    f'Average error from data: {order_of_magnitude.convert(np.round(np.mean(data - peaks_opt), 2), scale="mega")[0]} MOhm ({np.round(absolute_mean_percent_error(data, peaks_opt), 2)} %)')
print(
    f'Average error from previous model: {order_of_magnitude.convert(np.round(np.mean(peaks_gt - peaks_opt), 2), scale="mega")[0]} MOhm ({np.round(absolute_mean_percent_error(peaks_gt, peaks_opt), 2)} %)')

print("Value of x after long initial SET pulse:", np.max(x_gt[int(initial_time / model['dt']):]))
