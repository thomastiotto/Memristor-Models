import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import json

from functions import *
from order_of_magnitude import order_of_magnitude
from scipy.optimize import curve_fit

# p100mv_old = np.loadtxt('/Users/thomas/Desktop/+0.1V.csv', delimiter=',', usecols=1)
p100mv_old = np.loadtxt('../../../raw_data/pulses/m4V_10x_positive_pulse_p100mV-1V_m1V_measure_resistance.txt',
                        usecols=10, skiprows=2)[10:]
p100mv = np.loadtxt(
    "../../../raw_data/pulses/hold_p1V_10x_negative pulse_m4V_10x_positive_pulse_p100mV_m1V_measure.txt",
    delimiter="\t", skiprows=1, usecols=[1])[10:]
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
iterations = 10


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
for i in range(iterations):
    n.append(((r_pl[-1] - r_min) / r_max) ** (1 / c))
    r_pl.append(r_min + r_max * (n[-1] + 1) ** c)
n.append(((r_pl[-1] - r_min) / r_max) ** (1 / c))

############# YAKOPCIC MODEL ##############
readV = -1
p100readv = np.loadtxt(
    "../../../raw_data/pulses/hold_p1V_10x_negative pulse_m2V-m3V-m4V_10x_positive_pulse_p100mV-1V_steps_of_100mV_m1V_measure_3x.txt",
    delimiter="\t", skiprows=1, usecols=[0, 2])
read_length = read_pulse_length(p100readv, readV)

time, voltage, i, r, x = model_sim_with_params(0.001, -8.135891404816215, 10, 3.86621037038006, iterations, readV,
                                               0.001, **model)
r_yk = find_peaks(r, voltage, readV)[10:]
fig_plot_opt_debug = plot_images(time, voltage, i, r, x, f'-2 V / +0.1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True, consider_from=120000)
fig_plot_opt_debug.show()

############# PLOT #######################

fig, ax = plt.subplots()
x = np.arange(0, iterations, 1)
ax.plot(x, func(x, *popt_new), label='Fit to new data')
ax.plot(x, func(x, *popt_old), label='Fit to old data')
ax.plot(r_pl, label='Power-law')
ax.plot(r_yk, label='Yakopcic')
ax.plot(p100mv_old, label='Old data')
ax.plot(p100mv, label='New data')

plt.title('Memristor resistance')
ax.set_xlabel(r"Pulse number $n$")
ax.set_ylabel(r"Resistance $R (\Omega)$")
# ax.set_yscale("log")
ax.set_ylim([0, 2e8])
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
