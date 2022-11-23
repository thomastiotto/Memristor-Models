from yakopcic_functions import *
from yakopcic_model import *
from experiment_setup import *
from scipy import interpolate
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default='input.txt', help="File containing the voltage pulses to simulate.")
args = parser.parse_args()

# the model that works for pulses does not wortk for I-V
model = {'An': 0.02662694665,
         'Ap': 0.49699999999999994,
         'Vn': 0,
         'Vp': 0,
         'alphan': 0.7013461469,
         'alphap': 9.2,
         'bmax_n': 0.04662714454744514,
         'bmax_p': 4.988561168,
         'bmin_n': 0.044341659769156175,
         'bmin_p': 0.002125127287,
         'dt': 0.001,
         'eta': 1,
         'gmax_n': 5.117466299437892e-07,
         'gmax_p': 0.0004338454236,
         'gmin_n': 1.6806214980624974e-07,
         'gmin_p': 0.03135053798,
         'x0': 0.0,
         'xn': 0.1433673316,
         'xp': 0.11}
model = Memristor_Alina
sim_mode = 0

if sim_mode == 0:
    ### Main implementation.
    iptVs = startup2(input_iv)
    time, v = interactive_iv(iptVs, model['dt'])
    ###

if sim_mode == 1:
    ### Hardcoded IV pulse to better reproduce it.
    v = np.concatenate((
        np.linspace(0, 1, round(10 * 1 / dt_nengo)),
        np.linspace(1, -2, round(25 * 1 / dt_nengo)),
        np.linspace(-2, 0, round(15 * 1 / dt_nengo))))
    time = np.linspace(0, 50, len(v))
    ###

if sim_mode == 2:
    ### Using interpolation to produce the IV curve. Requires the "-2V_0.csv" file.
    data = pd.read_csv("../../imported_data/data/Radius 10 um/-2V_0.csv")
    time = np.array(data["Smu1.Time[1][1]"])
    v = np.array(data["Smu1.V[1][1]"])

    time_interp = np.linspace(0, time[-1], int((time[-1] - time[0]) / model["dt"]) + 1)
    volt_interp = interpolate.splrep(time, v, s=0, k=1)
    v = interpolate.splev(time_interp, volt_interp, der=0)
    time = time_interp
    ###

x = np.zeros(v.shape, dtype=float)
print("t: ", time.shape)
print(time[-1])

x[0] = model['x0']
for k in range(1, len(x)):
    x[k] = x[k - 1] + dxdt(v[k], x[k - 1], model['Ap'], model['An'], model['Vp'], model['Vn'], model['xp'],
                           model['xn'], model['alphap'], model['alphan'], 1) * model['dt']
    if x[k] < 0:
        x[k] = 0
    if x[k] > 1:
        x[k] = 1

i = current(v, x, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'],
            model['bmin_p'], model['gmin_n'], model['bmin_n'])
r = np.divide(v, i, out=np.zeros(v.shape, dtype=float), where=i != 0)

fig_plot = plot_images(time, v, i, r, x, f'-2 V / +1 V', plot_type='iv')
fig_plot.show()
fig_debug = plot_images(time, v, i, r, x, f'-2 V / +1 V', plot_type='debug', model=model)
fig_debug.show()


def find_nearest(array, value, range=None, exclude=None):
    import copy

    array_copy = copy.deepcopy(array)

    if exclude is not None:
        array_copy[exclude] = np.inf
    if range is not None:
        array_copy[:range[0]] = np.inf
        array_copy[range[1]:] = np.inf
    else:
        array_copy = array_copy

    idx = (np.abs(array_copy - value)).argmin()
    return idx


# -- find values of current at -1 V
readV = -1
cur_1 = current(readV, x[find_nearest(v, readV, (20000, 35000))], model['gmax_p'], model['bmax_p'], model['gmax_n'],
                model['bmax_n'],
                model['gmin_p'],
                model['bmin_p'], model['gmin_n'], model['bmin_n'])
print('Current at -1 V', cur_1, 'resistance', readV / cur_1, 'x', x[find_nearest(v, readV, (20000, 35000))])
cur_2 = current(readV, x[find_nearest(v, readV, (35000, -1))], model['gmax_p'], model['bmax_p'], model['gmax_n'],
                model['bmax_n'],
                model['gmin_p'],
                model['bmin_p'], model['gmin_n'], model['bmin_n'])
print('Current at -1 V', cur_2, 'resistance', readV / cur_2, 'x', x[find_nearest(v, readV, (35000, -1))])
cur_3 = current(readV, 0.582182200516887, model['gmax_p'], model['bmax_p'], model['gmax_n'],
                model['bmax_n'],
                model['gmin_p'],
                model['bmin_p'], model['gmin_n'], model['bmin_n'])
print('Current at -1 V', cur_3, 'resistance', readV / cur_3, 'x', 0.582182200516887)

# 10e-10 is noise floor
