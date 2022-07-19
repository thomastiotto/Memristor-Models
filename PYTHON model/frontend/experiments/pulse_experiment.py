import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import yakopcic_model
from experiment_setup import *
from functions import *
from yakopcic_model import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default='input.txt', help="File containing the IV curve")
args = parser.parse_args()

experiment = YakopcicSET()
I = experiment.functions["I"]
dxdt = experiment.functions["dxdt"]
dt = experiment.simulation["dt"]


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


def main():
    np.seterr(all="raise")
    iptVs = startup2()
    time, voltage = interactive_iv(iptVs, dt)
    x = solver2(dxdt, time, dt, 0.5, voltage)
    i = I(time, voltage, x)
    r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float) + 200, where=i != 0)

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


if __name__ == "__main__":
    main()
