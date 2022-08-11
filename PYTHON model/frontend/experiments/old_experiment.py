import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import yakopcic_functions
import yakopcic_model
from experiment_setup import *
from functions import *
from yakopcic_model import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default='input_iv.txt', help="File containing the voltage pulses to simulate.")
parser.add_argument("-type", "--plot_type", default=0, help="1: Regular resistance plot. 0: IV-plot.")
parser.add_argument("-d", "--debug", default=True, help="Show debug plots.")
args = parser.parse_args()
plot_type = args.plot_type

experiment = OldYakopcic()

I = experiment.functions["I"]
dxdt = experiment.functions["dxdt"]
dt = experiment.simulation["dt"]
memr = experiment.memristor if args.debug is True else None


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
    plot_images(args.file, plot_type, time, voltage, i, r, x, memr)


if __name__ == "__main__":
    main()
