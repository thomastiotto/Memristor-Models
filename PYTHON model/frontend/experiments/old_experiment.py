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
parser.add_argument("-f", "--file", default='input.txt', help="File containing the voltage pulses to simulate.")
parser.add_argument("-type", "--plot_type", default=1, help="1: Regular resistance plot. 0: IV-plot.")
args = parser.parse_args()
plot_type = args.plot_type

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
    print("t: ", len(time), "v: ", len(voltage))
    x = solver2(dxdt, time, dt, 0, voltage)

    i = I(time, voltage, x)
    r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float) + 200, where=i != 0)

    # Finds the indices that are representing local peaks.
    # Then, make a list of local peaks using the incides.
    peak_ids = scipy.signal.find_peaks(r)
    peak = [0]
    for idx in peak_ids[0]:
        peak.append(r[idx])

    if plot_type == 1: # Plots regular resistance plot; Full plot + its local peaks.
        fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 5))
        if args.file == "input.txt":
            ax1.plot(time[120000:], r[120000:]) # Supposes a 120s SET pulse!
            ax1.twinx().plot(time[120000:], voltage[120000:], color='r') # Voltage.
        else:
            ax1.plot(time, r) # Solution otherwise.
            ax1.twinx().plot(time, voltage, color='r')  # Voltage.
        ax1.set_yscale("log")
        ax2.plot(peak, "o")
        ax2.set_yscale("log")

    else: # Plots the IV curve.
        fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 5))
        ax1.plot(time, r)
        ax1.twinx().plot(time, voltage, color='r')
        ax2.plot(voltage, i)

    plt.show()


if __name__ == "__main__":
    main()
