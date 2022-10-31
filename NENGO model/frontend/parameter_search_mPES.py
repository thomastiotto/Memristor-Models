import argparse
import sys
from subprocess import run

from extras import *

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parameter", choices=["exponent", "noise", "neurons", "gain"], required=True)
parser.add_argument("-f", "--function", default="x")
parser.add_argument("-D", "--dimensions", default=3, type=int)
parser.add_argument("-N", "--neurons", type=int, default=10)
parser.add_argument("-i", "--inputs", default=["sine", "sine"], nargs="*", choices=["sine", "white"])
parser.add_argument("-l", "--limits", nargs=2, type=float, required=True)
parser.add_argument("-n", "--number", type=int, default=None)
parser.add_argument("-a", "--averaging", type=int, required=True)
parser.add_argument("-d", "--directory", default="../data/")
args = parser.parse_args()
# parameters to search
function = args.function
dimensions = args.dimensions
neurons = args.neurons
inputs = args.inputs
parameter = args.parameter
start_par = args.limits[0]
end_par = args.limits[1]
num_par = args.number if args.number is not None else end_par - start_par + 1
num_averaging = args.averaging
directory = args.directory

dir_name, dir_images, dir_data = make_timestamped_dir(root=directory + "parameter_search/" + str(parameter))
print("Reserved folder", dir_name)

res_list = np.linspace(start_par, end_par, num=num_par) if args.parameter in ["exponent", "noise", "neurons"] \
    else np.logspace(np.rint(start_par).astype(int), np.rint(end_par).astype(int),
                     num=np.rint(num_par).astype(int))
num_parameters = len(res_list)
print("Evaluation for", parameter, "with", neurons, "neurons")
print(f"Search limits of parameters: [{start_par},{end_par}]")
print("Number of parameters:", num_parameters)
print("Averaging per parameter", num_averaging)
print("Total iterations", num_parameters * num_averaging)

mse_list = []
spearman_list = []
mse_to_rho_list = []
number_set_pulses_list = []
number_reset_pulses_list = []
set_pulse_train_length_list = []
reset_pulse_train_length_list = []
counter = 0
for k, par in enumerate(res_list):
    print(f"Parameter #{k} ({par})")
    it_res_mse = []
    it_res_spearman = []
    it_res_mse_to_rho = []
    it_res_number_set_pulses = []
    it_res_number_reset_pulses = []
    it_res_set_pulse_train_length = []
    it_res_reset_pulse_train_length = []
    for avg in range(num_averaging):
        counter += 1
        print(f"[{counter}/{num_parameters * num_averaging}] Averaging #{avg + 1}")
        if parameter == "exponent":
            result = run(
                [sys.executable, "mPES.py", "--verbosity", str(1), "-P", str(par), "-N", str(neurons), "-f",
                 str(function), "-D", str(dimensions)]
                + ["-i"] + inputs,
                capture_output=True,
                universal_newlines=True)
        if parameter == "noise":
            result = run(
                [sys.executable, "mPES.py", "--verbosity", str(1), "-n", str(par), "-N", str(neurons), "-f",
                 str(function), "-D", str(dimensions)]
                + ["-i"] + inputs,
                capture_output=True,
                universal_newlines=True)
        if parameter == "neurons":
            rounded_neurons = str(np.rint(par).astype(int))
            result = run(
                [sys.executable, "mPES.py", "--verbosity", str(1), "-N", str(100), rounded_neurons, str(100), "-N",
                 str(neurons), "-f", str(function), "-D", str(dimensions)]
                + ["-i"] + inputs,
                capture_output=True,
                universal_newlines=True)
        if parameter == "gain":
            result = run(
                [sys.executable, "mPES.py", "--verbosity", str(1), "-g", str(par), "-f", str(function),
                 "-D", str(dimensions), "-N", str(neurons)]
                + ["-i"] + inputs,
                capture_output=True,
                universal_newlines=True)
        # save statistics
        try:
            mse = np.mean([float(i) for i in result.stdout.split("\n")[0][1:-1].split(",")])
            print("MSE", mse)
            it_res_mse.append(mse)
            spearman = np.mean([float(i) for i in result.stdout.split("\n")[1][1:-1].split(",")])
            print("Spearman", spearman)
            it_res_spearman.append(spearman)
            mse_to_rho = np.mean([float(i) for i in result.stdout.split("\n")[2][1:-1].split(",")])
            print("MSE-to-rho", mse_to_rho)
            it_res_mse_to_rho.append(mse_to_rho)

            number_set_pulses = np.mean([float(i) for i in result.stdout.split("\n")[3].split(",")])
            print("Number of SET pulses", number_set_pulses)
            it_res_number_set_pulses.append(number_set_pulses)
            number_reset_pulses = np.mean([float(i) for i in result.stdout.split("\n")[4].split(",")])
            print("Number of RESET pulses", number_reset_pulses)
            it_res_number_reset_pulses.append(number_reset_pulses)
            set_pulse_train_length = np.mean([float(i) for i in result.stdout.split("\n")[5].split(",")])
            print("SET pulse train length", set_pulse_train_length)
            it_res_set_pulse_train_length.append(set_pulse_train_length)
            reset_pulse_train_length = np.mean([float(i) for i in result.stdout.split("\n")[6].split(",")])
            print("RESET pulse train length", reset_pulse_train_length)
            it_res_reset_pulse_train_length.append(reset_pulse_train_length)
        except:
            print("Ret", result.returncode)
            print("Out", result.stdout)
            print("Err", result.stderr)
    mse_list.append(it_res_mse)
    spearman_list.append(it_res_spearman)
    mse_to_rho_list.append(it_res_mse_to_rho)
    number_set_pulses_list.append(it_res_number_set_pulses)
    number_reset_pulses_list.append(it_res_number_reset_pulses)
    set_pulse_train_length_list.append(it_res_set_pulse_train_length)
    reset_pulse_train_length_list.append(it_res_reset_pulse_train_length)

mse_means = np.mean(mse_list, axis=1)
spearman_means = np.mean(spearman_list, axis=1)
mse_to_rho_means = np.mean(mse_to_rho_list, axis=1)
number_set_pulses_means = np.mean(number_set_pulses_list, axis=1)
number_reset_pulses_means = np.mean(number_reset_pulses_list, axis=1)
set_pulse_train_length_means = np.mean(set_pulse_train_length_list, axis=1)
reset_pulse_train_length_means = np.mean(reset_pulse_train_length_list, axis=1)
print("Average MSE for each parameter:", mse_means)
print("Average Spearman for each parameter:", spearman_means)
print("Average MSE-to-rho for each parameter:", mse_to_rho_means)
print("Average number of SET pulses for each parameter:", number_set_pulses_means)
print("Average number of RESET pulses for each parameter:", number_reset_pulses_means)
print("Average SET pulse train length for each parameter:", set_pulse_train_length_means)
print("Average RESET pulse train length for each parameter:", reset_pulse_train_length_means)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, mse_means, label="MSE")
ax.legend()
fig.savefig(dir_images + "mse" + ".pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, spearman_means, label="Spearman")
ax.legend()
fig.savefig(dir_images + "correlations" + ".pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, mse_to_rho_means, label=r"$\frac{\rho}{\mathrm{MSE}}$")
ax.legend()
fig.savefig(dir_images + "mse-to-rho" + ".pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, number_set_pulses_means, label="Number of SET pulses")
ax.plot(res_list, number_reset_pulses_means, label="Number of RESET pulses")
ax.legend()
fig.savefig(dir_images + "number-pulses" + ".pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, set_pulse_train_length_means, label="SET pulse train length")
ax.plot(res_list, reset_pulse_train_length_means, label="RESET pulse train length")
ax.legend()
fig.savefig(dir_images + "pulse-train-length" + ".pdf")

print(f"Saved plots in {dir_images}")

np.savetxt(dir_data + "results.csv",
           np.stack((res_list, mse_means, spearman_means, mse_to_rho_means), axis=1),
           delimiter=",",
           header=parameter + ",MSE,Pearson,Spearman,Kendall,MSE-to-rho,SET pulses,RESET pulses,SET train,RESET train",
           comments="")
with open(dir_data + "parameters.txt", "w") as f:
    f.write(f"Parameter: {parameter}\n")
    f.write(f"Function: {function}\n")
    f.write(f"Dimensions: {dimensions}\n")
    f.write(f"Neurons: {neurons}\n")
    f.write(f"Input: {inputs}\n")
    f.write(f"Limits: [{start_par},{end_par}]\n")
    f.write(f"Number of searched parameters: {num_par}\n")
    f.write(f"Number of runs for averaging: {num_averaging}\n")
print(f"Saved data in {dir_data}")
