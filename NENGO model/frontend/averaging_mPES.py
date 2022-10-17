import argparse
import sys
from subprocess import run
from extras import *

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--averaging", default=100, type=int)
parser.add_argument("-i", "--inputs", default=["white", "sine"], nargs="*", choices=["sine", "white"])
parser.add_argument("-f", "--function", default="x")
parser.add_argument("-N", "--neurons", type=int, default=100)
parser.add_argument("-D", "--dimensions", type=int, default=3)
parser.add_argument("-g", "--gain", type=float, default=10e4)
parser.add_argument("-l", "--learning_rule", default="mPES", choices=["mPES", "PES"])
parser.add_argument("--directory", default="../data/")
parser.add_argument("-lt", "--learn_time", default=3 / 4, type=float)
parser.add_argument("-d", "--device", default="/cpu:0")
args = parser.parse_args()

learning_rule = args.learning_rule
gain = args.gain
function = args.function
inputs = args.inputs
neurons = args.neurons
dimensions = args.dimensions
num_averaging = args.averaging
directory = args.directory
learn_time = args.learn_time
device = args.device

dir_name, dir_images, dir_data = make_timestamped_dir(
    root=directory + "averaging/" + str(learning_rule) + "/" + "_" + str(inputs) + "_"
         + str(neurons) + "_" + str(dimensions) + "_" + str(gain))
print("Reserved folder", dir_name)

print("Evaluation for", learning_rule)
print("Averaging runs", num_averaging)

res_mse = []
res_spearman = []
res_mse_to_rho = []
res_number_set_pulses = []
res_number_reset_pulses = []
res_set_pulse_train_length = []
res_reset_pulse_train_length = []
counter = 0
for avg in range(num_averaging):
    counter += 1
    print(f"[{counter}/{num_averaging}] Averaging #{avg + 1}")
    result = run(
        [sys.executable, "mPES.py", "--verbosity", str(1), "-D", str(dimensions), "-l", str(learning_rule),
         "-N", str(neurons), "-f", str(function), "-lt", str(learn_time), "-g", str(gain),
         "-d", str(device)]
        + ["-i"] + inputs,
        capture_output=True,
        universal_newlines=True)

    # save statistics
    try:
        mse = np.mean([float(i) for i in result.stdout.split("\n")[0][1:-1].split(",")])
        print("MSE", mse)
        res_mse.append(mse)
        spearman = np.mean([float(i) for i in result.stdout.split("\n")[1][1:-1].split(",")])
        print("Spearman", spearman)
        res_spearman.append(spearman)
        mse_to_rho = np.mean([float(i) for i in result.stdout.split("\n")[2][1:-1].split(",")])
        print("MSE-to-rho", mse_to_rho)
        res_mse_to_rho.append(mse_to_rho)

        number_set_pulses = np.mean([float(i) for i in result.stdout.split("\n")[3].split(",")])
        print("Number of SET pulses", number_set_pulses)
        res_number_set_pulses.append(number_set_pulses)
        number_reset_pulses = np.mean([float(i) for i in result.stdout.split("\n")[4].split(",")])
        print("Number of RESET pulses", number_reset_pulses)
        res_number_reset_pulses.append(number_reset_pulses)
        set_pulse_train_length = np.mean([float(i) for i in result.stdout.split("\n")[5].split(",")])
        print("SET pulse train length", set_pulse_train_length)
        res_set_pulse_train_length.append(set_pulse_train_length)
        reset_pulse_train_length = np.mean([float(i) for i in result.stdout.split("\n")[6].split(",")])
        print("RESET pulse train length", reset_pulse_train_length)
        res_reset_pulse_train_length.append(reset_pulse_train_length)

    except:
        print("Ret", result.returncode)
        print("Out", result.stdout)
        print("Err", result.stderr)
mse_means = np.mean(res_mse)
spearman_means = np.mean(res_spearman)
mse_to_rho_means = np.mean(res_mse_to_rho)
print("Average MSE:", mse_means)
print("Average Spearman:", spearman_means)
print("Average MSE-to-rho:", mse_to_rho_means)
number_set_pulses_means = np.mean(res_number_set_pulses)
print("Average number of SET pulses:", number_set_pulses_means)
number_reset_pulses_means = np.mean(res_number_reset_pulses)
print("Average number of RESET pulses:", number_reset_pulses_means)
set_pulse_train_length_means = np.mean(res_set_pulse_train_length)
print("Average SET pulse train length:", set_pulse_train_length_means)
reset_pulse_train_length_means = np.mean(res_reset_pulse_train_length)
print("Average RESET pulse train length:", reset_pulse_train_length_means)

res_list = range(num_averaging)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, res_mse, label="MSE")
ax.legend()
fig.savefig(dir_images + "mse" + ".pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, res_spearman, label="Spearman")
ax.legend()
fig.savefig(dir_images + "correlations" + ".pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, res_number_set_pulses, label="Number of SET pulses")
ax.plot(res_list, res_number_reset_pulses, label="Number of RESET pulses")
ax.legend()
fig.savefig(dir_images + "number-pulses" + ".pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, res_set_pulse_train_length, label="SET pulse train length")
ax.plot(res_list, res_reset_pulse_train_length, label="RESET pulse train length")
ax.legend()
fig.savefig(dir_images + "pulse-train-length" + ".pdf")

print(f"Saved plots in {dir_images}")

np.savetxt(dir_data + "results.csv",
           np.stack((res_mse, res_spearman, res_mse_to_rho, res_number_set_pulses, res_number_reset_pulses,
                     res_set_pulse_train_length, res_reset_pulse_train_length), axis=1),
           delimiter=",",
           header="MSE,Pearson,Spearman,Kendall,MSE-to-rho,SET pulses,RESET pulses,SET train,RESET train", comments="")
with open(dir_data + "parameters.txt", "w") as f:
    f.write(f"Learning rule: {learning_rule}\n")
    f.write(f"Function: {function}\n")
    f.write(f"Neurons: {neurons}\n")
    f.write(f"Dimensions: {dimensions}\n")
    f.write(f"Number of runs for averaging: {num_averaging}\n")
print(f"Saved data in {dir_data}")
