import argparse
import time

from nengo.learning_rules import PES
from nengo.params import Default
from nengo.processes import WhiteSignal
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from extras import *
from yakopcic_learning import mPES

setup()

# Should not be useful for NengoDL>=3.3.0
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_control_flow_v2()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--function", default="x",
                    help="The function to learn.  Default is x")
parser.add_argument("-i", "--inputs", default=["sine", "sine"], nargs="*", choices=["sine", "white"],
                    help="The input signals [learning, testing].  Default is sine")
parser.add_argument("-t", "--timestep", default=0.001, type=int)
parser.add_argument("-S", "--simulation_time", default=30, type=int)
parser.add_argument("-N", "--neurons", nargs="*", default=[100], action="store", type=int,
                    help="The number of neurons used in the Ensembles [pre, post, error].  Default is 10")
parser.add_argument("-D", "--dimensions", default=3, type=int,
                    help="The number of dimensions of the input signal")
parser.add_argument("-n", "--noise", default=0.15, type=float,
                    help="The noise on the simulated memristors.  Default is 0.15")
parser.add_argument("-g", "--gain", default=1e6, type=float)  # default chosen by parameter search experiments
parser.add_argument("-l", "--learning_rule", default="mPES", choices=["mPES", "PES"])
parser.add_argument("-P", "--parameters", default=Default, type=float,
                    help="The parameters of simualted memristors.  For now only the exponent c")
parser.add_argument("-b", "--backend", default="nengo_core", choices=["nengo_dl", "nengo_core"])
parser.add_argument("-o", "--optimisations", default="run", choices=["run", "build", "memory"])
parser.add_argument("-s", "--seed", default=None, type=int)  # Can use seed = 13 for quick check
parser.add_argument("--plot", default=0, choices=[0, 1, 2, 3], type=int,
                    help="0: No visual output, 1: Show plots, 2: Save plots, 3: Save data")
parser.add_argument("--verbosity", default=2, choices=[0, 1, 2], type=int,
                    help="0: No textual output, 1: Only numbers, 2: Full output")
parser.add_argument("-pd", "--plots_directory", default="../data/",
                    help="Directory where plots will be saved.  Default is ../data/")
parser.add_argument("-d", "--device", default="/cpu:0",
                    help="/cpu:0 or /gpu:[x]")
parser.add_argument("-lt", "--learn_time", default=3 / 4, type=float)
parser.add_argument('--probe', default=1, choices=[0, 1, 2], type=int,
                    help="0: probing disabled, 1: only probes to calculate statistics, 2: all probes active")

args = parser.parse_args()
seed = args.seed
tf.random.set_seed(seed)
np.random.seed(seed)
function_string = "lambda x: " + args.function
function_to_learn = eval(function_string)
if len(args.inputs) not in (1, 2):
    parser.error('Either give no values for action, or two, not {}.'.format(len(args.inputs)))
if len(args.inputs) == 1:
    if args.inputs[0] == "sine":
        input_function_train = input_function_test = Sines(period=4)
    if args.inputs[0] == "white":
        input_function_train = input_function_test = WhiteSignal(period=60, high=5, seed=seed)
if len(args.inputs) == 2:
    if args.inputs[0] == "sine":
        input_function_train = Sines(period=4)
    if args.inputs[0] == "white":
        input_function_train = WhiteSignal(period=60, high=5, seed=seed)
    if args.inputs[1] == "sine":
        input_function_test = Sines(period=4)
    if args.inputs[1] == "white":
        input_function_test = WhiteSignal(period=60, high=5, seed=seed)
timestep = args.timestep
sim_time = args.simulation_time
if len(args.neurons) not in range(1, 3):
    parser.error('Either give no values for action, or one, or three, not {}.'.format(len(args.neurons)))
if len(args.neurons) == 1:
    pre_n_neurons = post_n_neurons = error_n_neurons = args.neurons[0]
if len(args.neurons) == 2:
    pre_n_neurons = error_n_neurons = args.neurons[0]
    post_n_neurons = args.neurons[1]
if len(args.neurons) == 3:
    pre_n_neurons = args.neurons[0]
    post_n_neurons = args.neurons[1]
    error_n_neurons = args.neurons[2]
dimensions = args.dimensions
noise_percent = args.noise
gain = args.gain
exponent = args.parameters
learning_rule = args.learning_rule
backend = args.backend
optimisations = args.optimisations
progress_bar = False
printlv1 = printlv2 = lambda *a, **k: None
if args.verbosity >= 1:
    printlv1 = print
if args.verbosity >= 2:
    printlv2 = print
    progress_bar = True
plots_directory = args.plots_directory
device = args.device
probe = args.probe
generate_plots = show_plots = save_plots = save_data = False
if args.plot >= 1:
    generate_plots = True
    show_plots = True
    probe = 2
if args.plot >= 2:
    save_plots = True
if args.plot >= 3:
    save_data = True

debug = True

# TODO give better names to folders or make hierarchy
if save_plots or save_data:
    dir_name, dir_images, dir_data = make_timestamped_dir(root=plots_directory + learning_rule + "/")

learn_time = int(sim_time * args.learn_time)
n_neurons = np.amax([pre_n_neurons, post_n_neurons])
if optimisations == "build":
    optimize = False
    sample_every = timestep
    simulation_discretisation = 1
elif optimisations == "run":
    optimize = True
    sample_every = timestep
    simulation_discretisation = 1
elif optimisations == "memory":
    optimize = False
    sample_every = timestep * 100
    simulation_discretisation = n_neurons
printlv2(f"Using {optimisations} optimisation")

model = nengo.Network(seed=seed)
with model:
    nengo_dl.configure_settings(inference_only=True)
    # Create an input node
    input_node = nengo.Node(
        output=SwitchInputs(input_function_train,
                            input_function_test,
                            switch_time=learn_time),
        size_out=dimensions
    )

    # Shut off learning by inhibiting the error population
    stop_learning = nengo.Node(output=lambda t: t >= learn_time)

    # Create the ensemble to represent the input, the learned output, and the error
    pre = nengo.Ensemble(pre_n_neurons, dimensions=dimensions, seed=seed)
    post = nengo.Ensemble(post_n_neurons, dimensions=dimensions, seed=seed)
    error = nengo.Ensemble(error_n_neurons, dimensions=dimensions, radius=2, seed=seed)

    # Connect pre and post with a communication channel
    # the matrix given to transform is the initial weights found in model.sig[conn]["weights"]
    # the initial transform has not influence on learning because it is overwritten by mPES
    # the only influence is on the very first timesteps, before the error becomes large enough
    conn = nengo.Connection(
        pre.neurons,
        post.neurons,
        transform=np.zeros((post.n_neurons, pre.n_neurons))
    )

    # Apply the learning rule to conn
    if learning_rule == "mPES":
        conn.learning_rule_type = mPES(
            noisy=noise_percent,
            gain=gain,
            seed=seed, )
    if learning_rule == "PES":
        conn.learning_rule_type = PES()
    printlv2("Simulating with", conn.learning_rule_type)

    # Provide an error signal to the learning rule
    nengo.Connection(error, conn.learning_rule)

    # Compute the error signal (error = actual - target)
    nengo.Connection(post, error)

    # Subtract the target (this would normally come from some external system)
    nengo.Connection(pre, error, function=function_to_learn, transform=-1)

    # Connect the input node to ensemble pre
    nengo.Connection(input_node, pre)

    nengo.Connection(
        stop_learning,
        error.neurons,
        transform=-20 * np.ones((error.n_neurons, 1)))

    # essential ones are used to calculate the statistics
    if probe > 0:
        pre_probe = nengo.Probe(pre, synapse=0.01, sample_every=sample_every)
        post_probe = nengo.Probe(post, synapse=0.01, sample_every=sample_every)
    if probe > 1:
        input_node_probe = nengo.Probe(input_node, sample_every=sample_every)
        error_probe = nengo.Probe(error, synapse=0.01, sample_every=sample_every)
        learn_probe = nengo.Probe(stop_learning, synapse=None, sample_every=sample_every)
        weight_probe = nengo.Probe(conn, "weights", synapse=None, sample_every=sample_every)
        post_spikes_probe = nengo.Probe(post.neurons, sample_every=sample_every)
        if isinstance(conn.learning_rule_type, mPES):
            x_pos_probe = nengo.Probe(conn.learning_rule, "x_pos", synapse=None,
                                      sample_every=sample_every)
            x_neg_probe = nengo.Probe(conn.learning_rule, "x_neg", synapse=None,
                                      sample_every=sample_every)

# Create the Simulator and run it
printlv2(f"Backend is {backend}, running on ", end="")
if backend == "nengo_core":
    printlv2("CPU")
    cm = nengo.Simulator(model, seed=seed, dt=timestep, optimize=optimize, progress_bar=progress_bar)
if backend == "nengo_dl":
    printlv2(device)
    cm = nengo_dl.Simulator(model, seed=seed, dt=timestep, progress_bar=progress_bar, device=device)
start_time = time.time()
with cm as sim:
    for i in range(simulation_discretisation):
        printlv2(f"\nRunning discretised step {i + 1} of {simulation_discretisation}")
        sim.run(sim_time / simulation_discretisation)
printlv2(f"\nTotal time for simulation: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))} s")

if isinstance(conn.learning_rule_type, mPES):
    # -- evaluate number of memristor pulses over simulation
    mpes_op = get_operator_from_sim(sim, 'SimmPES')

if probe > 0:
    # essential statistics
    y_true = sim.data[pre_probe][int((learn_time / timestep) / (sample_every / timestep)):, ...]
    y_pred = sim.data[post_probe][int((learn_time / timestep) / (sample_every / timestep)):, ...]
    # MSE after learning
    printlv2("MSE after learning [f(pre) vs. post]:")
    mse = mean_squared_error(function_to_learn(y_true), y_pred, multioutput='raw_values')
    printlv1(mse.tolist())
    # Correlation coefficients after learning
    correlation_coefficients = correlations(function_to_learn(y_true), y_pred)
    printlv2("Spearman correlation after learning [f(pre) vs. post]:")
    printlv1(correlation_coefficients[1])
    printlv2("MSE-to-rho after learning [f(pre) vs. post]:")
    printlv1(mse_to_rho_ratio(mse, correlation_coefficients[1]))

    if isinstance(conn.learning_rule_type, mPES) and debug:
        # -- evaluate number of memristor pulses over simulation
        pos_pulse_counter = mpes_op.pos_pulse_counter
        neg_pulse_counter = mpes_op.neg_pulse_counter
        printlv2('Average number of SET pulses')
        printlv1(np.mean(pos_pulse_counter))
        printlv2('Average number of RESET pulses')
        printlv1(np.mean(neg_pulse_counter))

        # -- evaluate the average length of consecutive reset or set pulses
        from itertools import groupby, product

        pulse_archive = np.array(mpes_op.pulse_archive)
        lengths_set = []
        lengths_reset = []
        for i, j in tqdm(product(range(pulse_archive.shape[1]), range(pulse_archive.shape[2])),
                         total=pulse_archive.shape[1] * pulse_archive.shape[2],
                         desc='Calculating average number of consecutive pulses'):
            for k, g in groupby(pulse_archive[:, i, j]):
                if k == 1:
                    lengths_set.append(len(list(g)))
                elif k == -1:
                    lengths_reset.append(len(list(g)))
        printlv2('Average length of consecutive SET pulses')
        printlv1(np.mean(lengths_set))
        printlv2('Average length of consecutive RESET pulses')
        printlv1(np.mean(lengths_reset))

if probe > 1:
    # Average
    printlv2("Weights average after learning:")
    printlv1(np.average(sim.data[weight_probe][-1, ...]))

    # Sparsity
    printlv2("Weights sparsity at t=0 and after learning:")
    printlv1(gini(sim.data[weight_probe][0]), end=" -> ")
    printlv1(gini(sim.data[weight_probe][-1]))

plots = {}
if generate_plots and probe > 1:
    plotter = Plotter(sim.trange(sample_every=sample_every), post_n_neurons, pre_n_neurons, dimensions,
                      learn_time,
                      sample_every,
                      plot_size=(13, 7),
                      dpi=300,
                      pre_alpha=0.3
                      )
    plots["results_smooth"] = plotter.plot_results(sim.data[input_node_probe], sim.data[pre_probe],
                                                   sim.data[post_probe],
                                                   error=
                                                   sim.data[post_probe] -
                                                   function_to_learn(sim.data[pre_probe]),
                                                   smooth=True)
    plots["results"] = plotter.plot_results(sim.data[input_node_probe], sim.data[pre_probe],
                                            sim.data[post_probe],
                                            error=
                                            sim.data[post_probe] -
                                            function_to_learn(sim.data[pre_probe]),
                                            smooth=False)
    plots["post_spikes"] = plotter.plot_ensemble_spikes("Post", sim.data[post_spikes_probe],
                                                        sim.data[post_probe])
    plots["weights"] = plotter.plot_weight_matrices_over_time(sim.data[weight_probe], sample_every=sample_every)

    plots["testing_smooth"] = plotter.plot_testing(function_to_learn(sim.data[pre_probe]),
                                                   sim.data[post_probe],
                                                   smooth=True)
    plots["testing"] = plotter.plot_testing(function_to_learn(sim.data[pre_probe]), sim.data[post_probe],
                                            smooth=False)
    if debug:
        res_pos, res_neg = mpes_op.compute_resistance(sim.data[x_pos_probe], sim.data[x_neg_probe])
    if n_neurons <= 10 and learning_rule == "mPES":
        # plots["weights_mpes"] = plotter.plot_weights_over_time(sim.data[x_pos_probe],
        #
        #                                                        sim.data[x_neg_probe])
        res_pos, res_neg = mpes_op.compute_resistance(sim.data[x_pos_probe], sim.data[x_neg_probe])
        plots["memristors"] = plotter.plot_values_over_time(res_pos, res_neg, value="resistance")

if save_plots:
    assert generate_plots and probe > 1

    for fig in plots.values():
        fig.savefig(dir_images + str(i) + ".pdf")
        # fig.savefig( dir_images + str( i ) + ".png" )

    print(f"Saved plots in {dir_images}")

if save_data:
    save_weights(dir_data, sim.data[weight_probe])
    print(f"Saved NumPy weights in {dir_data}")

    save_results_to_csv(dir_data, sim.data[input_node_probe], sim.data[pre_probe], sim.data[post_probe],
                        sim.data[post_probe] - function_to_learn(sim.data[pre_probe]))
    save_memristors_to_csv(dir_data, sim.data[x_pos_probe], sim.data[x_neg_probe])
    print(f"Saved data in {dir_data}")

if show_plots:
    assert generate_plots and probe > 1

    for fig in plots.values():
        fig.show()

    # DEBUG: zoom in on one synapse
    if learning_rule == "mPES" and debug:
        # -- weights trajectory
        plt.plot(np.mean(sim.data[weight_probe], axis=(1, 2)))
        plt.title("Average weight over time")
        plt.show()

        plt.figure(figsize=(20, 20))
        plt.plot(res_pos[4000:7500, 0, 0], c='r')
        plt.plot(res_neg[4000:7500, 0, 0], c='b')
        plt.show()
