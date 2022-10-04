from scipy import optimize
import json

from functions import *
from yakopcic_model import *
from tqdm.auto import tqdm

# -- model found with pulse_experiment_match_magnitude.py and pulse_experiment_finetuning.py
model = json.load(open('../../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an'))

readV = -1
debug = False


def model_sim_with_params(pulse_length, resetV, setV, readV, read_length, **params):
    input_pulses = set_pulse(resetV, setV, pulse_length, readV, read_length)
    iptVs = startup2(input_pulses)

    time, voltage = interactive_iv(iptVs, params['dt'])
    x = np.zeros(voltage.shape, dtype=float)

    for j in tqdm(range(1, len(x))):
        x[j] = x[j - 1] + dxdt(voltage[j], x[j - 1], params['Ap'], params['An'], params['Vp'], params['Vn'],
                               params['xp'],
                               params['xn'], params['alphap'], params['alphan'], 1) * params['dt']
        if x[j] < 0:
            x[j] = 0
        if x[j] > 1:
            x[j] = 1

    i = current(voltage, x,
                params['gmax_p'], params['bmax_p'], params['gmax_n'], params['bmax_n'],
                params['gmin_p'], params['bmin_p'], params['gmin_n'], params['bmin_n'])
    r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)

    return time, voltage, i, r, x


def residuals(x, peaks_gt, pulse_length, readV, read_length, model):
    resetV, setV = x

    time, voltage, i, r, x = model_sim_with_params(pulse_length, resetV, setV, readV, read_length, **model)
    peaks_model = find_peaks(r, voltage, readV)

    # print('Residual absolute error:', np.sum(np.abs(peaks_gt - peaks_model)))

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
times = []
for i in range(len(p100readv)):
    if p100readv[i, 1] == -1:
        times.append(p100readv[i + 1, 0] - p100readv[i, 0])
read_length = np.mean(times)
print('Average readV pulse length', np.round(read_length, 2), 'seconds')

print('------------------ VARIOUS SET V ------------------')
fig_plot_default, ax_plot = plt.subplots(1, 1, figsize=(6, 5))
ax_plot.plot(p100mv, 'o', label='+0.1 V (data)')
# ax_plot.plot(p500mv, 'o', label='+0.5 V (data)')
# ax_plot.plot(p1v, 'o', label='+1 V (data)')
ax_plot.set_prop_cycle(None)

Vset = [0.1]
for vi, v in enumerate(Vset):
    time, voltage, i, r, x = model_sim_with_params(1, -2, v, readV, read_length, **model)
    fig_plot_default = plot_images(time, voltage, i, r, x, f'-2 V / +{v} V (model)', readV, fig_plot_default)

fig_plot_default.show()

fig_plot_opt_debug = plot_images(time, voltage, i, r, x, f'-2 V / +0.1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True)
fig_plot_opt_debug.show()

# -- define ground truth
print('------------------ ORIGINAL ------------------')
time_gt, voltage_gt, i_gt, r_gt, x_gt = model_sim_with_params(1, -2, 1, readV, read_length, **model)
peaks_gt = find_peaks(r_gt, voltage_gt, readV)
fig_plot_opt_debug = plot_images(time_gt, voltage_gt, i_gt, r_gt, x_gt, f'-2 V / +0.1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True)
fig_plot_opt_debug.show()

# -- run optimisation
print('------------------ OPTIMISATION ------------------')
bounds = ([-20, 0.1], [-2, 20])
x0 = [bounds[1][0], bounds[0][1]]
res_minimisation = optimize.least_squares(residuals, x0, args=[p100mv, 0.001, readV, 0.001, model], bounds=bounds,
                                          method='dogbox', verbose=2)
print(f'Optimisation result:\nVreset: {res_minimisation.x[0]}\nVset: {res_minimisation.x[1]}')

# TODO find x value
print("Value of x after long initial SET pulse", )

# -- plot results
fig_plot_opt, ax_plot = plt.subplots(1, 1, figsize=(6, 5))
ax_plot.plot(p100mv, 'o', fillstyle='none', label='Data')

time, voltage, i, r, x = model_sim_with_params(0.001, res_minimisation.x[0], res_minimisation.x[1], readV, 0.001,
                                               **model)
fig_plot_opt = plot_images(time_gt, voltage_gt, i_gt, r_gt, x_gt, f'Model (1 s)', readV,
                           fig_plot_opt)
fig_plot_opt = plot_images(time, voltage, i, r, x, f'Model (1 ms)', readV,
                           fig_plot_opt)
fig_plot_opt.show()
fig_plot_opt_debug = plot_images(time, voltage, i, r, x, f'-2 V / +0.1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True)
fig_plot_opt_debug.show()
