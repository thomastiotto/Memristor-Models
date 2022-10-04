from scipy import optimize
import pprint
import copy
import json

from functions import *
from yakopcic_model import *
from tqdm.auto import tqdm

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


def model_sim_with_params(pulse_length, resetV, setV, readV, **params):
    input_pulses = set_pulse(resetV, setV, pulse_length, readV)
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


def model_fit_electron(x, pulse_length, readV, read_length):
    print(x)
    gmin_n, bmin_n = x
    resetV, setV = -2, 0.1

    input_pulses = set_pulse(resetV, setV, pulse_length, readV, read_length)
    iptVs = startup2(input_pulses)

    time, voltage = interactive_iv(iptVs, model['dt'])
    x = np.zeros(voltage.shape, dtype=float)

    for j in tqdm(range(1, len(x))):
        x[j] = x[j - 1] + dxdt(voltage[j], x[j - 1], model['Ap'], model['An'], model['Vp'], model['Vn'], model['xp'],
                               model['xn'], model['alphap'], model['alphan'], 1) * model['dt']
        if x[j] < 0:
            x[j] = 0
        if x[j] > 1:
            x[j] = 1

    i = current(voltage, x,
                model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'],
                model['gmin_p'], model['bmin_p'], gmin_n, bmin_n)
    r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)

    return time, voltage, i, r, x


def residuals_model_electron(x, peaks_gt, readV, read_length):
    time, voltage, i, r, x = model_fit_electron(x, 1, readV, read_length)
    peaks_model = find_peaks(r, voltage, readV)

    # print('Residual absolute error:', np.sum(np.abs(peaks_gt - peaks_model)))

    return peaks_gt - peaks_model


bounds = (-np.inf, np.inf)
x0 = [model['gmin_n'], model['bmin_n']]
res_minimisation_electron = optimize.least_squares(residuals_model_electron, x0, args=[p100mv, readV, read_length],
                                                   bounds=bounds,
                                                   method='lm', verbose=2)
print(f'Electron transfer regression result:\n'
      f'gmax_n: {model["gmin_n"]} → {res_minimisation_electron.x[0]}\n'
      f'bmax_n: {model["bmin_n"]} → {res_minimisation_electron.x[1]}\n')

time, voltage, i, r, x = model_fit_electron(res_minimisation_electron.x, 1, readV, read_length)
fig_plot_fit_electron, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(p100mv, 'o', label='Data')
fig_plot_fit_electron = plot_images(time, voltage, i, r, x, f'Model', readV,
                                    fig_plot_fit_electron)
fig_plot_fit_electron.show()
fig_plot_fit_electron_debug = plot_images(time, voltage, i, r, x, f'Model', readV,
                                          fig_plot_fit_electron,
                                          plot_type='debug', model=model, show_peaks=True)
fig_plot_fit_electron_debug.show()

model_upd = copy.deepcopy(model)
model_upd['gmin_n'] = res_minimisation_electron.x[0]
model_upd['bmin_n'] = res_minimisation_electron.x[1]

peaks_model = find_peaks(r, voltage, readV)
print('Average error:', np.mean(p100mv - peaks_model))
pprint.pprint(model_upd)
json.dump(model_upd, open('../../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an', 'w'), indent=2)
