import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

from fit_model_to_data_functions import *

iterations = 10

# p100mv_old = np.loadtxt('/Users/thomas/Desktop/+0.1V.csv', delimiter=',', usecols=1)
p100mv_old = np.loadtxt(
    '../../../raw_data/pulses/old_device/m4V_10x_positive_pulse_p100mV-1V_m1V_measure_resistance.txt',
    usecols=10, skiprows=2)[10:]
if iterations < len(p100mv_old):
    p100mv_old = p100mv_old[:iterations]
p100mv = np.loadtxt(
    "../../../raw_data/pulses/old_device/hold_p1V_10x_negative pulse_m4V_10x_positive_pulse_p100mV_m1V_measure.txt",
    delimiter="\t", skiprows=1, usecols=[1])[9:]
if iterations < len(p100mv):
    p100mv = p100mv[:iterations]


def func(x, a, b, c):
    return a + b * x ** c


x = np.arange(0, len(p100mv), 1)
popt_new, _ = curve_fit(func, x, p100mv, maxfev=1000000)

x = np.arange(0, len(p100mv_old), 1)
popt_old, _ = curve_fit(func, x, p100mv_old, maxfev=1000000)

############ POWER-LAW MODEL ############

r_min = 2e2
r_max = 2.3e8
c = -0.146

n = []
r_pl = [p100mv_old[0]]
for i in range(iterations - 1):
    n.append(((r_pl[-1] - r_min) / r_max) ** (1 / c))
    r_pl.append(r_min + r_max * (n[-1] + 1) ** c)
n.append(((r_pl[-1] - r_min) / r_max) ** (1 / c))

############# YAKOPCIC MODEL OLD DATA ##############
model = json.load(open('../../../fitted/fitting_pulses/old_device/regress_negative_xp_alphap-adjusted_ap_an'))
readV = -1
p100readv = np.loadtxt(
    "../../../raw_data/pulses/old_device/hold_p1V_10x_negative pulse_m2V-m3V-m4V_10x_positive_pulse_p100mV-1V_steps_of_100mV_m1V_measure_3x.txt",
    delimiter="\t", skiprows=1, usecols=[0, 2])

read_length = read_pulse_length(p100readv, readV)

time, voltage, i, r, x = model_sim_with_params(pulse_length=0.001,
                                               resetV=-8.135891404816215, numreset=10,
                                               setV=3.86621037038006, numset=iterations,
                                               readV=readV, read_length=0.001,
                                               init_set_length=120, init_setV=1,
                                               **model)
r_yk_old = find_peaks(r, voltage, readV, 120)[9:-1]
fig_plot_opt_debug = plot_images(time, voltage, i, r, x, f'-2 V / +0.1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True, consider_from=120)
fig_plot_opt_debug.show()

############# YAKOPCIC MODEL NEWEST DATA ##############
model = json.load(open('../../../fitted/fitting_pulses/new_device/regress_negative_then_positive'))
model['dt'] = 0.0003
readV = -0.5
new_data = np.loadtxt('../../../raw_data/pulses/new_device/Sp1V_RSm2V_Rm500mV_processed.txt', delimiter='\t',
                      skiprows=1,
                      usecols=[2])[99:]
if iterations < len(new_data):
    new_data = new_data[:iterations]

new_data = readV / new_data

time, voltage, i, r, x = model_sim_with_params(pulse_length=0.0007,
                                               resetV=-6.342839121380956, numreset=100,
                                               setV=4.8838587394343485, numset=iterations,
                                               readV=readV, read_length=0.0003,
                                               init_set_length=60, init_setV=4.8838587394343485,
                                               **model)
r_yk_new = find_peaks(r, voltage, readV, 60, dt=model['dt'])[99:-1]
fig_plot_opt_debug = plot_images(time, voltage, i, r, x, f'-2 V / +1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True,
                                 consider_from=60, dt=model['dt'])
fig_plot_opt_debug.show()

############# PLOT #######################

fig, ax = plt.subplots()
x = np.arange(0, iterations, 1)
# ax.plot(x, func(x, *popt_new), label='Fit to new data')
# ax.plot(x, func(x, *popt_old), label='Fit to old data')
ax.plot(r_pl, label='Power-law')
ax.plot(r_yk_old, label='Yakopcic old')
ax.plot(r_yk_new, label='Yakopcic new')
ax.plot(p100mv_old, label='Oldest data')
ax.plot(p100mv, label='Old data')
ax.plot(new_data, label='Newest data')

plt.title('Memristor resistance')
ax.set_xlabel(r"Pulse number $n$")
ax.set_ylabel(r"Resistance $R (\Omega)$")
ax.set_yscale("log")
# ax.set_ylim([0, 2e8])
ax.legend(loc='best')
ax.tick_params(axis='x')
ax.tick_params(axis='y')
plt.tight_layout()
fig.show()

# print("Difference in resistances:")
# print(np.abs(np.array(p100mv) - np.array(r)))
# print("Average difference in resistances:")
# avg_err = np.sum(np.abs(np.array(p100mv) - np.array(r))) / len(r)
# print(avg_err, f"({(avg_err * 100) / (r_max - r_min)} %)")
# print("MSE in resistances:")
# print(mse(p100mv, r))
