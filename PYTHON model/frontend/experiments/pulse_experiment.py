from scipy import optimize

from experiment_setup import *
from functions import *
from yakopcic_model import *
from tqdm.auto import tqdm

model = Memristor_Alina


def model_sim(x, pulse_length):
    # scipy expects a 1d array
    resetV, setV = x

    input_pulses = set_pulse(resetV, setV, pulse_length, -1)
    iptVs, ipds, num_waves = startup2(input_pulses)

    time, voltage = interactive_iv(iptVs, model['dt'])
    x = np.zeros(voltage.shape, dtype=float)

    for j in tqdm(range(1, len(x))):
        x[j] = x[j - 1] + dxdt(voltage[j], x[j - 1], model['Ap'], model['An'], model['Vp'], model['Vn'], model['xp'],
                               model['xn'], model['alphap'], model['alphan'], 1) * model['dt']
        if x[j] < 0:
            x[j] = 0
        if x[j] > 1:
            x[j] = 1

    i = current(voltage, x, model['gmax_p'], model['bmax_p'], model['gmax_n'], model['bmax_n'], model['gmin_p'],
                model['bmin_p'],
                model['gmin_n'], model['bmin_n'])
    r = np.divide(voltage, i, out=np.zeros(voltage.shape, dtype=float), where=i != 0)

    return time, voltage, i, r, x, ipds, num_waves


def residuals(x, peaks_gt):
    time, voltage, i, r, x, ipds, num_waves = model_sim(x, 0.001)
    peaks_model = find_peaks(r, ipd=ipds[2], num_peaks=num_waves[1] + num_waves[2])

    print('Residual absolute error:', np.sum(np.abs(peaks_gt - peaks_model)))

    return peaks_gt - peaks_model


print('------------------ VARIOUS SET V ------------------')
Vset = [0.1, 0.5, 1]
for vi, v in enumerate(Vset):
    time, voltage, i, r, x, ipds, num_waves = model_sim((-4, v), 1)
    if vi == 0:
        fig_plot_orig, fig_debug_orig = plot_images(time, voltage, i, r, x, ipds[2], num_waves[1] + num_waves[2],
                                                    f'+{v} V', model=model, debug=True)
    else:
        fig_plot_orig, fig_debug_orig = plot_images(time, voltage, i, r, x, ipds[2], num_waves[1] + num_waves[2],
                                                    f'+{v} V',
                                                    fig_plot_orig, model=model, debug=True)

fig_plot_orig.show()

# -- define ground truth
print('------------------ ORIGINAL ------------------')
time_gt, voltage_gt, i_gt, r_gt, x_gt, ipds_gt, num_waves_gt = model_sim((-4, 0.1), 1)
peaks_gt = find_peaks(r_gt, ipd=ipds_gt[2], num_peaks=num_waves_gt[1] + num_waves_gt[2])
# r_gt = find_peaks(r)

# -- run optimisation
print('------------------ OPTIMISATION ------------------')
bounds = ([-20, 1], [-4, 20])
x0 = [bounds[1][0], bounds[0][1]]
res_minimisation = optimize.least_squares(residuals, x0, args=[peaks_gt], bounds=bounds,
                                          method='dogbox', verbose=2)
print(f'DOGBOX result:\nVreset: {res_minimisation.x[0]}\nVset: {res_minimisation.x[1]}')

# -- plot results
time_opt, voltage_opt, i_opt, r_opt, x_opt, ipds_opt, num_waves_opt = model_sim(res_minimisation.x, 0.001)
fig_plot_opt, fig_debug_opt = plot_images(time_opt, voltage_opt, i_opt, r_opt, x_opt, ipds_opt[2],
                                          num_waves_opt[1] + num_waves_opt[2],
                                          f'RESET: {round(res_minimisation.x[0], 2)} V\nSET: {round(res_minimisation.x[1], 2)} V',
                                          model=model, debug=True)
fig_plot_opt, fig_debug_opt = plot_images(time_gt, voltage_gt, i_gt, r_gt, x_gt, ipds_gt[2],
                                          num_waves_gt[1] + num_waves_gt[2],
                                          f'RESET: {round(res_minimisation.x[0], 2)} V\nSET: {round(res_minimisation.x[1], 2)} V',
                                          fig_plot_opt, model=model, debug=True)
fig_plot_opt.show()

# TODO let's double check everything is ok with the pulses OK
# TODO let's see if we can even better approximate the real results OK
# TODO let's try different SET voltages like in Fig. 4B of my paper WIP
# TODO let's try the working model in Nengo (old+new experiments)
# TODO once we're happy, let's see what the internal state is after the RESET pulses and let's use that as initialisation in Nengo
# TODO extend mPES to RESET voltages
