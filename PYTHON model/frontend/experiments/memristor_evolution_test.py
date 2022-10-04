import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import json

from functions import *
from order_of_magnitude import order_of_magnitude

p100 = np.loadtxt('/Users/thomas/Desktop/+0.1V.csv', delimiter=',', usecols=1)

iterations = 10

############ POWER-LAW MODEL ############

r_min = 2e2
r_max = 2.3e8
c = -0.146

n = []
r_pl = [p100[0]]
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

model = json.load(open('../../../fitted/fitting_pulses/regress_negative_xp_alphap-adjusted_ap_an'))

time, voltage, i, r, x = model_sim_with_params(1, -2, 0.1, readV, read_length, **model)
r_yk = find_peaks(r, voltage, readV)[10:]
fig_plot_opt_debug = plot_images(time, voltage, i, r, x, f'-2 V / +0.1 V', readV,
                                 plot_type='debug', model=model, show_peaks=True)
fig_plot_opt_debug.show()

############# PLOT #######################

fig, ax = plt.subplots()
ax.plot(p100, 'o', label='Old data TBC')
ax.plot(r_pl, 'o', label='Power-law')
ax.plot(r_yk, 'o', label='Yakopcic')

plt.title('Memristor resistance')
ax.set_xlabel(r"Pulse number $n$")
ax.set_ylabel(r"Resistance $R (\Omega)$")
ax.set_yscale("log")
ax.legend(loc='best')
ax.tick_params(axis='x')
ax.tick_params(axis='y')
plt.tight_layout()
fig.show()

print("Difference in resistances:")
print(np.abs(np.array(p100) - np.array(r)))
print("Average difference in resistances:")
avg_err = np.sum(np.abs(np.array(p100) - np.array(r))) / len(r)
print(avg_err, f"({(avg_err * 100) / (r_max - r_min)} %)")
print("MSE in resistances:")
print(mse(p100, r))
