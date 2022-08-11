import argparse
import scipy.signal

from yakopcic_functions import *
from functions import *
from yakopcic_model import *
from experiment_setup import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default='input.txt', help="File containing the voltage pulses to simulate.")
parser.add_argument("-type", "--plot_type", default=1, help="1: Regular resistance plot. 0: IV-plot.")
parser.add_argument("-d", "--debug", default=True, help="Show debug plots.")
args = parser.parse_args()
plot_type = args.plot_type

experiment = NewYakopcic()
dt = experiment.simulation["dt"]
memr_debug = experiment.memristor if args.debug is True else None  # For debugging.
memr = experiment.memristor  # To keep referring to the parameters explicitly as this script does.
# dt = 0.0843  # Maybe Alina's


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
    x = np.zeros(voltage.shape, dtype=float)
    print("t: ", len(time), "v: ", len(voltage))

    for j in range(1, len(x)):
        x[j] = x[j - 1] + dxdt(voltage[j], x[j - 1], memr.Ap, memr.An, memr.Vp, memr.Vn, memr.xp,
                               memr.xn, memr.alphap, memr.alphan, 1) * dt
        if x[j] < 0:
            x[j] = 0
        if x[j] > 1:
            x[j] = 1
    i = current(voltage, x, memr.gmax_p, memr.bmax_p, memr.gmax_n, memr.bmax_n, memr.gmin_p, memr.bmin_p,
                memr.gmin_n, memr.bmin_n)
    r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)
    plot_images(args.file, plot_type, time, voltage, i, r, x, memr_debug)


if __name__ == "__main__":
    main()

#TODO why can't we reproduce Alina's behaviour exactly?
