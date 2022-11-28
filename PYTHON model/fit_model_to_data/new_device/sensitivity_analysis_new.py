from yakopcic_functions import *
from fit_model_to_data_functions import *
import pandas as pd
import json
import re


def load_model():
    """
    Load the model.
    Returns:
        model (dict).
    """
    # -- model found with pulse_experiment_match_magnitude.py and pulse_experiment_finetuning.py
    model = json.load(open('../../../fitted/fitting_pulses/new_device/regress_negative_then_positive'))

    # fig_plot_default = plot_images(time, voltage, i, r, x, f'{round(resetV, 3)} V / +{round(setV, 3)} V (model)',
    #                               readV, fig_plot_default)
    # fig_plot_default.show()
    # ax_plot.plot(p500mv, 'o', label='+0.5 V (data)')
    # ax_plot.plot(p1v, 'o', label='+1 V (data)')
    # ax_plot.set_prop_cycle(None)

    return model


def load_parameters(model):
    """
    Load the parameters from the model.
    Args:
        model: memristor model, dict.

    Returns:
        parameters: for I and dxdt, list.
        names: for I and dxdt, list.
        dataframe: w/ parameters.
        out: latex output.
    """
    area = 10
    params_dxdt = np.array([model["Ap"], model["An"], model["Vp"], model["Vn"], model["xp"], model["xn"],
                            model["alphap"], model["alphan"]])
    params_I_on = np.array([model["gmax_p"], model["bmax_p"], model["gmax_n"], model["bmax_n"]])
    params_I_off = np.array([model["gmin_p"], model["bmin_p"], model["gmin_n"], model["bmin_n"]])
    params_I = np.concatenate((params_I_on, params_I_off))
    params_all = np.concatenate((params_dxdt, params_I))

    param_names_dxdt = ["$A_p$", "$A_n$", "$V_p$", "$V_n$", "$x_p$", "$x_n$", "$\\alpha_p$", "$\\alpha_n$"]
    param_names_I = ["$g_{\max , p}$", "$b_{\max , p}$", "$g_{\max , n}$", "$b_{\max , n}$", "$g_{\min , p}$",
                     "$b_{\min , p}$", "$g_{\min , n}$", "$b_{\min , n}$"]

    df_params = pd.DataFrame()
    df_params_tmp = pd.DataFrame([params_all])
    df_params_tmp = df_params_tmp.transpose()
    df_params_tmp.columns = [f'Mean {area}']
    df_params_tmp.index = param_names_dxdt + param_names_I
    df_params = pd.concat([df_params, df_params_tmp], axis=1)

    out = df_params.to_latex(escape=False)
    out = re.sub(r"\d+\.\d{6}(?!\d)", lambda x: str(round(float(x.group(0)), 2)), out)
    out = re.sub(r'[0]+\.[0]+e-\d', '-', out)
    out = re.sub(r'\d+\.\d+e-\d', lambda x: f'\\num{{{x.group(0)}}}', out)

    return params_dxdt, params_I, param_names_dxdt, param_names_I, df_params, out, area


def find_sensitivity(sign, params, idx, case, model, name):
    """

    Args:
        sign: sign of change, positive or negative, int.
        params: parameters to change from depending on case, list.
        idx: index of parameter to change, int.
        case: case to change; |dxdt| or |I|, str.
        model: memristor model, dict.
        name: name of parameter to change, str.

    Returns:
        dif: the amount of change in a parameter required to achieve a 10% error, float.

    """

    # -- EXPERIMENT HYPERPARAMETNERS
    resetV = -5.846891601011591
    setV = 4.410540843557414
    readV = -0.5
    initialV = setV
    num_reset_pulses = 100
    num_set_pulses = 100
    nengo_time = 0.001
    nengo_program_time = nengo_time * 0.7
    nengo_read_time = nengo_time * 0.3
    initial_time = 60
    count_iter = 0
    dif = 50

    print("Calculating ground truth ({}): {} = {}".format(case, name, model["{}".format(name)]))
    time, current_v, current_i, current_r, current_x = \
        model_sim_with_params(pulse_length=nengo_program_time,
                              resetV=resetV, numreset=num_reset_pulses,
                              setV=setV, numset=num_set_pulses,
                              readV=readV, read_length=nengo_read_time,
                              init_set_length=initial_time, init_setV=initialV,
                              **model,
                              progress_bar=False)
    peaks_gt = find_peaks(current_r, current_v, readV, consider_from=initial_time, debug=False)
    plt.plot(range(len(peaks_gt)), peaks_gt)

    new_model = model.copy()
    gt_param = model["{}".format(name)]

    while count_iter < 500:  # Run until the error is about 10% or 500 iterations pass.
        if (name == "Vp" or name == "Vn") and (model["{}".format(name)] == 0):  # Skip unnecessary calculations.
            break

        new_model["{}".format(name)] = gt_param + sign * (dif * params[idx] / 100)
        if case == "I":
            print("Calculating new model (I), {} = {}:".format(
                name, gt_param + sign * (dif * params[idx] / 100)))
            time, voltage, i, r, x = model_sim_with_params(pulse_length=nengo_program_time,
                                                           resetV=resetV, numreset=num_reset_pulses,
                                                           setV=setV, numset=num_set_pulses,
                                                           readV=readV, read_length=nengo_read_time,
                                                           init_set_length=initial_time, init_setV=initialV,
                                                           **new_model,
                                                           progress_bar=False)
            peaks_new = find_peaks(r, voltage, readV=readV, consider_from=initial_time, debug=False)
        else:
            print("Calculating new model (dxdt), {} = {}:".format(
                name, gt_param + sign * (dif * params[idx] / 100)))
            time, voltage, i, r, x = model_sim_with_params(pulse_length=nengo_program_time,
                                                           resetV=resetV, numreset=num_reset_pulses,
                                                           setV=setV, numset=num_set_pulses,
                                                           readV=readV, read_length=nengo_read_time,
                                                           init_set_length=initial_time, init_setV=initialV,
                                                           **new_model,
                                                           progress_bar=False)
            peaks_new = find_peaks(r, voltage, readV=readV, consider_from=initial_time, debug=False)
            # print("New: ", new_current_sim)
            # print("Current I length:", len(this_current))
            # print("New I length:", len(new_current_sim))

        change = absolute_mean_percent_error(peaks_gt, peaks_new)
        # print("change:", change)
        plt.plot(range(len(peaks_new)), peaks_new)

        if 9.90 <= change <= 10.10 or dif <= 0:
            break
        print("change:", change)
        print("dif:", dif)

        if np.isnan(change):
            print("NAN for {} dif = {}".format(name, sign * (dif * params[idx] / 100)))
            dif = 404.404  # Error code | Overflow, etc.
            break

        if change > 10:
            dif = round(dif - 0.1, 1)
        elif change < 10:
            dif = round(dif + 10, 1) if change < 0.01 else round(dif + 0.1, 1)

        count_iter += 1

    shift_dir = "increasing" if sign == 1 else "decreasing"
    plt.title(f"{name} sensitivity | {shift_dir}")
    plt.show()

    return dif


def sensitivity_analysis(params_dxdt, params_I, param_names_dxdt, param_names_I, area, model):
    """

    Args:
        params_dxdt: dxdt parameters, list.
        params_I: I parameters, list.
        param_names_dxdt: dxdt parameter names, list.
        param_names_I: I parameter names, list.
        area: area of memristor; used for dataframe labelling, str.
        model: ground truth memristor model, dict.

    Returns:
        df_params: dataframe of parameter sensitivities, pd.DataFrame.

    """
    names_dxdt = ["Ap", "An", "Vp", "Vn", "xp", "xn", "alphap", "alphan"]
    names_I = ["gmax_p", "bmax_p", "gmax_n", "bmax_n", "gmin_p", "bmin_p", "gmin_n", "bmin_n"]
    df_results = pd.DataFrame()

    idxs = range(8)
    print('Area', area, 'um')
    diffs = []
    print("Decrease parameters by:")
    diff_decr = []
    for idx, names in zip(idxs, param_names_dxdt):
        diff = find_sensitivity(-1, params_dxdt, idx, "dxdt", model, names_dxdt[idx])
        diff_decr.append(diff)
        print(param_names_dxdt[idx] + ": " + str(diff) + "%")

    for idx, names in zip(idxs, param_names_I):
        diff = find_sensitivity(-1, params_I, idx, "I", model, names_I[idx])
        diff_decr.append(diff)
        print(param_names_I[idx] + ": " + str(diff) + "%")
    diffs.append(diff_decr)

    print("\nIncrease parameters by:")
    diff_incr = []
    for idx, names in zip(idxs, param_names_dxdt):
        diff = find_sensitivity(1, params_dxdt, idx, "dxdt", model, names_dxdt[idx])
        diff_incr.append(diff)
        print(param_names_dxdt[idx] + ": " + str(diff) + "%")

    for idx, names in zip(idxs, param_names_I):
        diff = find_sensitivity(1, params_I, idx, "I", model, names_I[idx])
        diff_incr.append(diff)
        print(param_names_I[idx] + ": " + str(diff) + "%")
    diffs.append(diff_incr)

    df2 = pd.DataFrame(diffs)
    df2 = df2.transpose()
    df2.columns = [f'Decrease \% ({area})', f'Increase \% ({area})']
    df2.index = param_names_dxdt + param_names_I
    df_results[[f'Decrease \% ({area})', f'Increase \% ({area})']] = df2

    return df_results


def df_latex(df_params, df_results):
    df_results['average'] = df_results.mean(axis=1).round(2)
    df_results.sort_values(by='average', inplace=True)
    print(df_results.to_latex(escape=False).replace('100.0', '\\textgreater 100.0'))
    print(df_results.drop(columns=['average']).to_latex(escape=False).replace('100.0', '\\textgreater 100.0'))

    out = df_params.to_latex(escape=False)
    print(df_params.to_latex(escape=False))


# Load the memristor model.
model = load_model()

# Extract the parameters, their names and make a dataframe.
params_dxdt, params_I, param_names_dxdt, param_names_I, df_params, output, area = load_parameters(model)

# Perform a sensitivity analysis, show resulting dataframe.
df_results = sensitivity_analysis(params_dxdt, params_I, param_names_dxdt, param_names_I, area, model)

# Plot results
plt.show()

# Convert dataframes to latex.
df_latex(df_params, df_results)

# TODO: Decide on the number of iterations, and the percentage change (how much should dif shift by?).
