from scipy import optimize
import pprint
import copy
import json

from functions import *

model = {
    "An": 0.02662694665 / 2,
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
    "gmin_n": 1.6115417930443725e-07,
    "bmin_n": 0.04644815038498206,
    "gmin_p": 0.03135053798,
    "x0": 0.0,
    "xn": 0.1433673316,
    "xp": 0.4810738043615987
}
readV = -1
debug = False

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


def residuals_model_electron(x, peaks_gt, pulse_length, readV, read_length, model):
    model_upd = copy.deepcopy(model)
    model_upd['gmin_n'] = x[0]
    model_upd['bmin_n'] = x[1]
    print('gmin_n:', x[0], 'bmin_n:', x[1])

    time, voltage, i, r, x = model_sim_with_params(pulse_length, -2, 10, 0.1, 10, readV, read_length, **model_upd)
    peaks_model = find_peaks(r, voltage, readV)

    return peaks_gt - peaks_model


bounds = (-np.inf, np.inf)
x0 = [model['gmin_n'], model['bmin_n']]
res_minimisation_electron = optimize.least_squares(residuals_model_electron, x0,
                                                   args=[p100mv, 1, readV, read_length, model],
                                                   bounds=bounds,
                                                   method='lm', verbose=2)
print(f'Electron transfer regression result:\n'
      f'gmax_n: {model["gmin_n"]} → {res_minimisation_electron.x[0]}\n'
      f'bmax_n: {model["bmin_n"]} → {res_minimisation_electron.x[1]}\n')
model_upd = copy.deepcopy(model)
model_upd['gmin_n'] = res_minimisation_electron.x[0]
model_upd['bmin_n'] = res_minimisation_electron.x[1]

time, voltage, i, r, x = model_sim_with_params(1, -2, 10, 0.1, 10, readV, read_length, **model_upd)
fig_plot_fit_electron, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(p100mv, 'o', label='Data')
fig_plot_fit_electron = plot_images(time, voltage, i, r, x, f'Model', readV,
                                    fig_plot_fit_electron)
fig_plot_fit_electron.show()
fig_plot_fit_electron_debug = plot_images(time, voltage, i, r, x, f'Model', readV,
                                          fig_plot_fit_electron,
                                          plot_type='debug', model=model, show_peaks=True)
fig_plot_fit_electron_debug.show()

peaks_model = find_peaks(r, voltage, readV)
print('Average error:', np.mean(p100mv - peaks_model))
pprint.pprint(model_upd)
json.dump(model_upd, open('../../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an', 'w'), indent=2)
