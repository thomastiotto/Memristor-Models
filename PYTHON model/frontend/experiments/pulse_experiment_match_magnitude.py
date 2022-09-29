from scipy import optimize
import pprint
import copy

from experiment_setup import *
from functions import *
from yakopcic_model import *
from tqdm.auto import tqdm

model = {'An': 0.02662694665,
         'Ap': 0.071 * 7,
         'Vn': 0,
         'Vp': 0,
         'alphan': 0.7013461469,
         'alphap': 9.2,
         'bmax_n': 6.272960721,
         'bmax_p': 4.988561168,
         'bmin_n': 3.295533935,
         'bmin_p': 0.002125127287,
         'dt': 0.001,
         'eta': 1,
         'gmax_n': 8.44e-06,
         'gmax_p': 0.0004338454236,
         'gmin_n': 1.45e-05,
         'gmin_p': 0.03135053798,
         'x0': 0.0,
         'xn': 0.1433673316,
         'xp': 0.11}
readV = -1
debug = False

pprint.pprint(model)


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


def model_sim(x, pulse_length, readV):
    # scipy expects a 1d array
    resetV, setV = x

    input_pulses = set_pulse(resetV, setV, pulse_length, readV)
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
                model['gmin_p'], model['bmin_p'], model['gmin_n'], model['bmin_n'])
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


def model_fit_electron(x, pulse_length, readV):
    print(x)
    gmax_n, bmax_n, gmin_n, bmin_n = x
    resetV, setV = -2, 0.1

    input_pulses = set_pulse(resetV, setV, pulse_length, readV)
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
                model['gmax_p'], model['bmax_p'], gmax_n, bmax_n,
                model['gmin_p'], model['bmin_p'], gmin_n, bmin_n)
    r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)

    return time, voltage, i, r, x


def residuals_model_electron(x, peaks_gt, readV):
    time, voltage, i, r, x = model_fit_electron(x, 1, readV)
    peaks_model = find_peaks(r, voltage, readV)

    # print('Residual absolute error:', np.sum(np.abs(peaks_gt - peaks_model)))

    return peaks_gt - peaks_model


bounds = (-np.inf, np.inf)
x0 = [model['gmax_n'] / 100, model['bmax_n'] / 100, model['gmin_n'] / 100, model['bmin_n'] / 100]
res_minimisation_electron = optimize.least_squares(residuals_model_electron, x0, args=[p100mv, readV], bounds=bounds,
                                                   method='lm', verbose=2)
print(f'Electron transfer regression result:\n'
      f'gmax_n: {model["gmax_n"]} → {res_minimisation_electron.x[0]}\n'
      f'bmax_n: {model["bmax_n"]} → {res_minimisation_electron.x[1]}\n'
      f'gmin_n: {model["gmin_n"]} → {res_minimisation_electron.x[2]}\n'
      f'bmin_n: {model["bmin_n"]} → {res_minimisation_electron.x[3]}\n')

time, voltage, i, r, x = model_fit_electron(res_minimisation_electron.x, 1, readV)
fig_plot_fit_electron, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
ax2.plot(p100mv, 'o')
fig_plot_fit_electron = plot_images(time, voltage, i, r, x, f'-2 V / +0.1 V', readV,
                                    fig_plot_fit_electron)
fig_plot_fit_electron.show()
fig_plot_fit_electron_debug = plot_images(time, voltage, i, r, x, f'-2 V / +0.1 V', readV,
                                          fig_plot_fit_electron,
                                          plot_type='debug', model=model, show_peaks=True)
fig_plot_fit_electron_debug.show()

model_upd = copy.deepcopy(model)
model_upd['gmax_n'] = res_minimisation_electron.x[0]
model_upd['bmax_n'] = res_minimisation_electron.x[1]
model_upd['gmin_n'] = res_minimisation_electron.x[2]
model_upd['bmin_n'] = res_minimisation_electron.x[3]

peaks_model = find_peaks(r, voltage, readV)
print('Average error:', np.mean(p100mv - peaks_model))
pprint.pprint(model_upd)
