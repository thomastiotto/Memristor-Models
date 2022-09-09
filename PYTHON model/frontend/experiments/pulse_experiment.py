from experiment_setup import *
from functions import *
from yakopcic_model import *

model = Memristor_Alina
iptVs = startup2(input_pulses)

time, voltage = interactive_iv(iptVs, model['dt'])
x = np.zeros(voltage.shape, dtype=float)
print("t: ", len(time), "v: ", len(voltage))

for j in range(1, len(x)):
    x[j] = x[j - 1] + dxdt(voltage[j], x[j - 1], model['Ap'], model['An'], model['Vp'], model['Vn'], model['xp'],
                            model['xn'], model['alphap'], model['alphan'], 1) * model['dt']
    if x[j] < 0:
        x[j] = 0
    if x[j] > 1:
        x[j] = 1

i = current(voltage, x, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'], model['bmin_p'],
            model['gmin_n'], model['bmin_n'])
r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)

plot_type=1
fig_plot, fig_debug = plot_images(plot_type, time, voltage, i, r, x, model)

# TODO let's double check everything is ok with the pulses
# TODO let's see if we can even better approximate the real results
# TODO let's try different SET voltages like in Fig. 4B of my paper
# TODO let's try the working model in Nengo (old+new experiments)
# TODO once we're happy, let's see what the internal state is after the RESET pulses and let's use that as initialisation in Nengo
# TODO extend mPES to RESET voltages