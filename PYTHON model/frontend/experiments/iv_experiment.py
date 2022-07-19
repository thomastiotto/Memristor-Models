import argparse

import scipy.signal

from yakopcic_functions import *
from functions import *
from yakopcic_model import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default='input_iv.txt', help="File containing the IV curve")
args = parser.parse_args()

# TODO why can't we reproduce Alina's behaviour exactly?
dt = 10e-3
dt = 0.0843  # Maybe Alina's

gmax_p = 0.0004338454236
bmax_p = 4.988561168
gmax_n = 8.44e-6
bmax_n = 6.272960721
gmin_p = 0.03135053798
bmin_p = 0.002125127287
gmin_n = 1.45e-05
bmin_n = 3.295533935
Ap = 0.071
An = 0.02662694665
Vp = 0
Vn = 0
xp = 0.11
xn = 0.1433673316
alphap = 9.2
alphan = 0.7013461469
xo = 0
eta = 1


def startup2():
    iptVs = {}
    with open(args.file, "r") as input_file:
        lines = input_file.readlines()

    wave_number = 1
    for line in lines:
        t_rise, t_on, t_fall, t_off, V_on, V_off, n_cycles = map(float, line.split())
        iptV = {"t_rise": t_rise, "t_on": t_on, "t_fall": t_fall, "t_off": t_off, "V_on": V_on, "V_off": V_off,
                "n_cycles": int(n_cycles)}
        iptVs["{}".format(wave_number)] = iptV
        wave_number += 1

    return iptVs


np.seterr(all="raise")
iptVs = startup2()
time, voltage = interactive_iv(iptVs, dt)
x = np.zeros(voltage.shape, dtype=float)

for j in range(1, len(x)):
    x[j] = x[j - 1] + dxdt(voltage[j], x[j - 1], Ap, An, Vp, Vn, xp, xn, alphap, alphan, 1) * dt
    if x[j] < 0:
        x[j] = 0
    if x[j] > 1:
        x[j] = 1
i = current(voltage, x, gmax_p, bmax_p, gmax_n, bmax_n, gmin_p, bmin_p, gmin_n, bmin_n)
r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)

peak_ids = scipy.signal.find_peaks(r)
peak = [0]
for idx in peak_ids[0]:
    peak.append(r[idx])

peak = peak[0:1] + peak[4:14] + peak[14::2] + peak[-1]

plt.figure(figsize=(7, 5))
plt.plot(range(0, len(peak)), peak, "o", markerfacecolor='none', ms=5, markeredgecolor='green')
# plt.title("Resistance of the Yakopcic memristor")
plt.xlabel("Pulse Number", fontsize=15)
plt.ylabel("Resistance (Ω)", fontsize=15)
plt.ylim(10e4, 2e6)
plt.yscale("log")
plt.xticks((0, 5, 10, 20))
plt.show()

fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 5))
ax1.plot(time, r)
ax1.twinx().plot(time, voltage, color='r')
ax2.plot(voltage, i)
plt.show()
