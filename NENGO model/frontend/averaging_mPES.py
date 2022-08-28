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
         + str(neurons) + "_" + str(dimensions) + "_" + str(gain) + "/")
print("Reserved folder", dir_name)

print("Evaluation for", learning_rule)
print("Averaging runs", num_averaging)

res_mse = []
res_pearson = []
res_spearman = []
res_kendall = []
res_mse_to_rho = []
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
        pearson = np.mean([float(i) for i in result.stdout.split("\n")[1][1:-1].split(",")])
        print("Pearson", pearson)
        res_pearson.append(pearson)
        spearman = np.mean([float(i) for i in result.stdout.split("\n")[2][1:-1].split(",")])
        print("Spearman", spearman)
        res_spearman.append(spearman)
        kendall = np.mean([float(i) for i in result.stdout.split("\n")[3][1:-1].split(",")])
        print("Kendall", kendall)
        res_kendall.append(kendall)
        mse_to_rho = np.mean([float(i) for i in result.stdout.split("\n")[4][1:-1].split(",")])
        print("MSE-to-rho", mse_to_rho)
        res_mse_to_rho.append(mse_to_rho)
    except:
        print("Ret", result.returncode)
        print("Out", result.stdout)
        print("Err", result.stderr)
mse_means = np.mean(res_mse)
pearson_means = np.mean(res_pearson)
spearman_means = np.mean(res_spearman)
kendall_means = np.mean(res_kendall)
mse_to_rho_means = np.mean(res_mse_to_rho)
print("Average MSE:", mse_means)
print("Average Pearson:", pearson_means)
print("Average Spearman:", spearman_means)
print("Average Kendall:", kendall_means)
print("Average MSE-to-rho:", mse_to_rho_means)

res_list = range(num_averaging)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, res_mse, label="MSE")
ax.legend()
fig.savefig(dir_images + "mse" + ".pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_list, res_pearson, label="Pearson")
ax.plot(res_list, res_spearman, label="Spearman")
ax.plot(res_list, res_kendall, label="Kendall")
ax.legend()
fig.savefig(dir_images + "correlations" + ".pdf")

print(f"Saved plots in {dir_images}")

np.savetxt(dir_data + "results.csv",
           np.stack((res_mse, res_pearson, res_spearman, res_kendall, res_mse_to_rho), axis=1),
           delimiter=",", header="MSE,Pearson,Spearman,Kendall,MSE-to-rho", comments="")
with open(dir_data + "parameters.txt", "w") as f:
    f.write(f"Learning rule: {learning_rule}\n")
    f.write(f"Function: {function}\n")
    f.write(f"Neurons: {neurons}\n")
    f.write(f"Dimensions: {dimensions}\n")
    f.write(f"Number of runs for averaging: {num_averaging}\n")
print(f"Saved data in {dir_data}")
