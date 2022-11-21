import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

p100mv = np.loadtxt(
    "../../../raw_data/pulses/old_device/hold_p1V_10x_negative pulse_m4V_10x_positive_pulse_p100mV_m1V_measure.txt",
    delimiter="\t", skiprows=1, usecols=[1])
p500mv = np.loadtxt(
    "../../../raw_data/pulses/old_device/hold_p1V_10x_negative pulse_m4V_10x_positive_pulse_p500mV_m1V_measure.txt",
    delimiter="\t", skiprows=1, usecols=[1])
p1v = np.loadtxt(
    "../../../raw_data/pulses/old_device/hold_p1V_10x_negative pulse_m4V_10x_positive_pulse_p1V_m1V_measure.txt",
    delimiter="\t", skiprows=1, usecols=[1])

plt.plot(p100mv, 'o')
plt.plot(p500mv, 'o')
plt.plot(p1v, 'o')
plt.yscale('log')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.show()
