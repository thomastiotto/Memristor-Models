import argparse
import pathlib
import sys
from subprocess import run

import pandas as pd

from extras import *

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parameter", choices=["noise", "gain"], required=True,
                    help="Parameter to vary [noise or gain]")
parser.add_argument("-e", "--experiment", type=int, default=1)
parser.add_argument("-l", "--limits", nargs=2, type=float, required=True,
                    help='The lower and upper limits for the parameter values')
parser.add_argument("-n", "--number", type=int, default=None, help='The number of parameter values to test')
parser.add_argument("-a", "--averaging", type=int, required=True, help='The number of times to average the results')
parser.add_argument("-d", "--directory", default="../data", help='The directory to save the results to')
args = parser.parse_args()

parameter = args.parameter
num_par = args.number if args.number is not None else args.limits[1] - args.limits[0] + 1
num_averaging = args.averaging

dir_name = make_timestamped_dir(root=args.directory + "/parameter_search/" + str(parameter))
print("Reserved folder", dir_name)

if parameter == 'gain' and np.any(np.array(args.limits) > 10):
    print('Assuming gain is in log scale, converting to linear')
    start_par = np.log10(args.limits[0])
    end_par = np.log10(args.limits[1])

res_list = np.linspace(start_par, end_par, num=num_par) if args.parameter in ["noise"] \
    else np.logspace(np.rint(start_par).astype(int), np.rint(end_par).astype(int),
                     num=np.rint(num_par).astype(int))
num_parameters = len(res_list)

print(f"Search limits of parameters: [{start_par},{end_par}]")
print("Number of parameters:", num_parameters)
print("Averaging per parameter", num_averaging)
print("Total iterations", num_parameters * num_averaging)

for k, par in enumerate(res_list):
    print(f"Parameter #{k} ({par})")
    print('----------------------------------------')
    if parameter == "noise":
        if np.any(np.array(args.limits) > 1.0):
            print('Assuming noise is given in %, converting to decimal')
            par /= 100

        result = run(
            [sys.executable, "learn_multidimensional_functions.py",
             '-E', str(args.experiment),
             '--directory', str(dir_name),
             '-I', str(num_averaging),
             '--no-hierarchy',
             '-g', str(1e6),
             '-n', str(par),
             ])
    elif parameter == "gain":
        result = run(
            [sys.executable, "learn_multidimensional_functions.py",
             '-E', str(args.experiment),
             '--directory', str(dir_name),
             '-I', str(num_averaging),
             '--no-hierarchy',
             '-g', str(par),
             '-n', str(0.15),
             ])

    # rename dirextory with parameter value
    newest_dir = max(pathlib.Path(dir_name).glob('*/'), key=os.path.getmtime)
    os.rename(newest_dir, os.path.join(dir_name, str(par)))

# -- read the results (error on last testing block) from the folders created by the experiments
df = pd.DataFrame()
for root, dirs, files in os.walk(dir_name):
    dirs.sort()
    if not dirs:
        df_temp = pd.read_csv(os.path.join(root, 'results.csv'), usecols=[0, 1, 2])
        df_temp = df_temp.tail(1)
        df_temp['parameter_value'] = float(root.split('/')[-1])
        print(float(root.split('/')[-1]))
        df = pd.concat([df, df_temp])
df.set_index('parameter_value', inplace=True)
df.to_csv(os.path.join(dir_name, 'results.csv'))
print('Saved results in', dir_name)
print('Simulation results:')
print(df)

# plot average testing error on last block
size_L = 10
size_M = 8
size_S = 6
x_label = parameter[0].upper() + parameter[1:]
plot_title = f'Performance vs. {x_label}'

fig, ax = plt.subplots()
fig.set_size_inches((3.5, 3.5 * ((5. ** 0.5 - 1.0) / 2.0)))
plt.tight_layout()
plt.title(plot_title, fontsize=size_L)
x = df.index.values
# x = (np.arange(num_testing_blocks + 1) * 2 * learn_block_time).astype(np.int)
ax.set_ylabel("Total error on last learning block", fontsize=size_M)
ax.set_xlabel(x_label, fontsize=size_M)
ax.tick_params(axis='x', labelsize=size_S)
ax.tick_params(axis='y', labelsize=size_S)
ax.plot(x, df['Mean mPES error'], label="Learned (mPES)", c="g")
ax.plot(x, df['CI mPES +'], linestyle="--", alpha=0.5, c="g")
ax.plot(x, df['CI mPES -'], linestyle="--", alpha=0.5, c="g")
ax.fill_between(x, df['CI mPES -'], df['CI mPES +'], alpha=0.3, color="g")
ax.legend(loc="best", fontsize=size_S)
fig.show()

fig.savefig(os.path.join(dir_name, 'results.pdf'))
print(f"Saved plots in {dir_name}")
