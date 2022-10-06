from scipy import optimize
import json
from order_of_magnitude import order_of_magnitude

from functions import *

# -- model found with pulse_experiment_match_magnitude.py and pulse_experiment_finetuning.py
model = json.load(open('../../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an'))
model = {
    "An": 0.013313473325,
    "Ap": 0.49699999999999994,
    "Vn": 0,
    "Vp": 0,
    "alphan": 0.7013461469,
    "alphap": 58.389867489576986,
    "bmax_n": 0.06683952060471841,
    "bmax_p": 4.988561168,
    "bmin_p": 0.002125127287,
    "dt": 0.001,
    "eta": 1,
    "gmax_n": 3.214000608914307e-07,
    "gmax_p": 0.0004338454236,
    "gmin_n": 8.045180116926906e-08,
    "bmin_n": 0.08304147962984593,
    "gmin_p": 0.03135053798,
    "x0": 0.0,
    "xn": 0.1433673316,
    "xp": 0.4810738043615987
}

readV = -1
debug = False


def residuals(x, peaks_gt, pulse_length, readV, read_length, model):
    resetV, setV = x

    time, voltage, i, r, x = model_sim_with_params(pulse_length, resetV, 10, setV, 10, readV, read_length, **model)
    peaks_model = find_peaks(r, voltage, readV)

    return peaks_gt - peaks_model


print('------------------ IMPORT REAL DATA ------------------')
p100mv = np.loadtxt(
    "../../../raw_data/pulses/hold_p1V_10x_negative pulse_m4V_10x_positive_pulse_p100mV_m1V_measure.txt",
    delimiter="\t", skiprows=1, usecols=[1])
p500mv = np.loadtxt(
    "../../../raw_data/pulses/hold_p1V_10x_negative pulse_m4V_10x_positive_pulse_p500mV_m1V_measure.txt",
    delimiter="\t", skiprows=1, usecols=[1])
p1v = np.loadtxt("../../../raw_data/pulses/hold_p1V_10x_negative pulse_m4V_10x_positive_pulse_p1V_m1V_measure.txt",
                 delimiter="\t", skiprows=1, usecols=[1])

# -- calculate the length of read pulses
p100readv = np.loadtxt(
    "../../../raw_data/pulses/hold_p1V_10x_negative pulse_m2V-m3V-m4V_10x_positive_pulse_p100mV-1V_steps_of_100mV_m1V_measure_3x.txt",
    delimiter="\t", skiprows=1, usecols=[0, 2])
read_length = read_pulse_length(p100readv, readV)

print('------------------ VARIOUS SET V ------------------')
fig_plot_default, ax_plot = plt.subplots(1, 1, figsize=(6, 5))
ax_plot.plot(p100mv, 'o', label='+0.1 V (data)')
# ax_plot.plot(p500mv, 'o', label='+0.5 V (data)')
# ax_plot.plot(p1v, 'o', label='+1 V (data)')
ax_plot.set_prop_cycle(None)

Vset = [0.1]
for vi, v in enumerate(Vset):
    time, voltage, i, r, x = model_sim_with_params(1, -2, 10, v, 10, readV, read_length, **model)
    fig_plot_default = plot_images(time, voltage, i, r, x, f'-2 V / +{v} V (model)', readV, fig_plot_default)

fig_plot_default.show()

# -- define ground truth
print('------------------ ORIGINAL ------------------')
time_gt, voltage_gt, i_gt, r_gt, x_gt = model_sim_with_params(1, -2, 10, 0.1, 10, readV, read_length, **model)
peaks_gt = find_peaks(r_gt, voltage_gt, readV)
fig_plot_opt_debug = plot_images(time_gt, voltage_gt, i_gt, r_gt, x_gt, f'-2 V / +0.1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True)
fig_plot_opt_debug.show()
print('Average error:', order_of_magnitude.convert(np.round(np.mean(p100mv - peaks_gt), 2), scale="mega")[0], 'MOhm')

# -- run optimisation
print('------------------ OPTIMISATION ------------------')
bounds = ([-20, 0.1], [-2, 20])
x0 = [bounds[1][0], bounds[0][1]]
res_minimisation = optimize.least_squares(residuals, x0, args=[peaks_gt, 0.001, readV, 0.001, model], bounds=bounds,
                                          method='dogbox', verbose=2)
print(f'Optimisation result:\nVreset: {res_minimisation.x[0]}\nVset: {res_minimisation.x[1]}')

# -- plot results
fig_plot_opt, ax_plot = plt.subplots(1, 1, figsize=(6, 5))
ax_plot.plot(p100mv, 'o', fillstyle='none', label='Data')

time, voltage, i, r, x = model_sim_with_params(0.001, res_minimisation.x[0], 10, res_minimisation.x[1], 10, readV,
                                               0.001,
                                               **model)
fig_plot_opt = plot_images(time_gt, voltage_gt, i_gt, r_gt, x_gt, f'Model (1 s)', readV,
                           fig_plot_opt)
fig_plot_opt = plot_images(time, voltage, i, r, x, f'Model (1 ms)', readV,
                           fig_plot_opt)
fig_plot_opt.show()
consider_from = int(120 / model['dt'])
fig_plot_opt_debug = plot_images(time, voltage, i, r, x, f'-2 V / +0.1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True, consider_from=consider_from)
fig_plot_opt_debug.show()

peaks_opt = find_peaks(r_gt, voltage_gt, readV)
print('Average error:', order_of_magnitude.convert(np.round(np.mean(p100mv - peaks_opt), 2), scale="mega")[0], 'MOhm')

print("Value of x after long initial SET pulse:", np.max(x_gt[consider_from:]))
