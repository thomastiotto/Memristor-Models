import glob
import atexit
import pickle
import re
import shutil
import signal

from matplotlib.offsetbox import AnchoredText
from order_of_magnitude import order_of_magnitude

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from nengo.learning_rules import PES
from nengo.processes import WhiteSignal

from extras import *
from yakopcic_learning_new import mPES


def cleanup(exit_code=None, frame=None):
    try:
        print("Cleaning up leftover files...")
        # delete tmp folder
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        pass


atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# make tmp folder with timestamp
tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(tmp_folder)


class Trevor_Estimator(BaseEstimator, RegressorMixin):
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __init__(self, experiment, learning_rule, seed=None,
                 initial_weights=None, low_memory=False):
        nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'
        nengo.rc['decoder_cache']['enabled'] = 'False'

        self.learning_rule = learning_rule
        self.low_memory = low_memory
        self.results_ready = False

        self.experiment = experiment
        self.initial_weights = initial_weights
        self.seed = seed

        if self.experiment == 1:
            self.exp_string = "PRODUCT experiment"
            self.exp_name = "Multiplying two numbers"
            self.function_to_learn = lambda x: x[0] * x[1]
            # [ pre, post, ground_truth, error ]
            self.neurons = [200, 200, 100, 100]
            self.dimensions = [2, 1, 1, 1]
            self.sim_time = 50
            self.img_name = 'product'
        if self.experiment == 2:
            self.exp_string = "COMBINED PRODUCTS experiment"
            self.exp_name = "Combining two products"
            self.function_to_learn = lambda x: x[0] * x[1] + x[2] * x[3]
            # [ pre, post, ground_truth, error ]
            self.neurons = [400, 400, 100, 100]
            self.dimensions = [4, 1, 1, 1]
            self.sim_time = 100
            self.img_name = 'combined_products'
        if self.experiment == 3:
            self.exp_string = "SEPARATE PRODUCTS experiment"
            self.exp_name = "Three separate products"
            self.function_to_learn = lambda x: [x[0] * x[1], x[0] * x[2], x[1] * x[2]]
            # [ pre, post, ground_truth, error ]
            self.neurons = [300, 300, 300, 300]
            self.dimensions = [3, 3, 3, 3]
            self.sim_time = 100
            self.img_name = 'separate_products'
        if self.experiment == 4:
            self.exp_string = "2D CIRCULAR CONVOLUTIONS experiment"
            self.exp_name = "Two-dimensional circular convolution"
            # [ pre, post, ground_truth, error, conv ]
            self.neurons = [400, 400, 200, 200, 200]
            self.dimensions = [4, 2, 2, 2, 2]
            self.function_to_learn = lambda x: np.fft.ifft(
                np.fft.fft(x[:int(self.dimensions[0] / 2)]) * np.fft.fft(x[int(self.dimensions[0] / 2):])
            )
            self.sim_time = 200
            self.img_name = '2d_cconv'
        if self.experiment == 5:
            self.exp_string = "3D CIRCULAR CONVOLUTIONS experiment"
            self.exp_name = "Three-dimensional circular convolution"
            # [ pre, post, ground_truth, error, conv ]
            self.neurons = [600, 300, 300, 300, 300]
            self.dimensions = [6, 3, 3, 3, 3]
            self.function_to_learn = lambda x: np.fft.ifft(
                np.fft.fft(x[:int(self.dimensions[0] / 2)]) * np.fft.fft(x[int(self.dimensions[0] / 2):])
            )
            self.sim_time = 400
            self.img_name = '3d_cconv'

        self.learn_block_time = 2.5
        # to have an extra testing block at t=[0,2.5]
        self.sim_time += self.learn_block_time
        convolve = False if experiment <= 3 else True

        self.num_blocks = int(self.sim_time / self.learn_block_time)
        self.num_testing_blocks = int(self.num_blocks / 2)

        self.timestep = 0.001
        self.strategy = 'symmetric-probabilistic'

        self.model = nengo.Network()
        with self.model:
            nengo_dl.configure_settings(stateful=False)

            self.model.inp = nengo.Node(
                # WhiteNoise( dist=Gaussian( 0, 0.05 ), seed=seed ),
                WhiteSignal(self.sim_time, high=5, seed=seed),
                size_out=self.dimensions[0]
            )
            self.model.pre = nengo.Ensemble(self.neurons[0], dimensions=self.dimensions[0], seed=seed)
            self.model.post = nengo.Ensemble(self.neurons[1], dimensions=self.dimensions[1], seed=seed)
            self.model.ground_truth = nengo.Ensemble(self.neurons[2], dimensions=self.dimensions[2], seed=seed)

            nengo.Connection(self.model.inp, self.model.pre)

            if convolve:
                self.model.conv = nengo.networks.CircularConvolution(self.neurons[4], self.dimensions[4], seed=seed)
                nengo.Connection(self.model.inp[:int(self.dimensions[0] / 2)],
                                 self.model.conv.input_a,
                                 synapse=None)
                nengo.Connection(self.model.inp[int(self.dimensions[0] / 2):],
                                 self.model.conv.input_b,
                                 synapse=None)
                nengo.Connection(self.model.conv.output, self.model.ground_truth,
                                 synapse=None)
            else:
                nengo.Connection(self.model.inp, self.model.ground_truth,
                                 function=self.function_to_learn,
                                 synapse=None)

    def fit(self, X, y=None, verbose=False):
        with self.model:
            if self.learning_rule:
                self.model.error = nengo.Ensemble(self.neurons[3], dimensions=self.dimensions[3], seed=seed)

                if isinstance(self.learning_rule, mPES):
                    self.model.conn = nengo.Connection(
                        self.model.pre.neurons,
                        self.model.post.neurons,
                        transform=np.zeros((self.model.post.n_neurons, self.model.pre.n_neurons)),
                        learning_rule_type=self.learning_rule
                    )

                elif isinstance(self.learning_rule, PES):
                    with open(f'{tmp_folder}/initial_weights_{self.seed}.npy', 'rb') as f:
                        initial_weights = np.load(f)

                    self.model.conn = nengo.Connection(
                        self.model.pre.neurons,
                        self.model.post.neurons,
                        transform=initial_weights,
                        learning_rule_type=self.learning_rule
                    )
                else:
                    self.model.conn = nengo.Connection(
                        self.model.pre,
                        self.model.post,
                        function=lambda x: np.random.random(self.dimensions[1]),
                        learning_rule_type=self.learning_rule
                    )
                nengo.Connection(self.model.error, self.model.conn.learning_rule)
                nengo.Connection(self.model.post, self.model.error)
                nengo.Connection(self.model.ground_truth, self.model.error, transform=-1)

                class cyclic_inhibit:
                    def __init__(self, cycle_time):
                        self.out_inhibit = 0.0
                        self.cycle_time = cycle_time

                    def step(self, t):
                        if t % self.cycle_time == 0:
                            if self.out_inhibit == 0.0:
                                self.out_inhibit = 2.0
                            else:
                                self.out_inhibit = 0.0

                        return self.out_inhibit

                self.model.inhib = nengo.Node(cyclic_inhibit(self.learn_block_time).step)
                nengo.Connection(self.model.inhib, self.model.error.neurons,
                                 transform=[[-1]] * self.model.error.n_neurons)
            else:
                self.model.conn = nengo.Connection(
                    self.model.pre,
                    self.model.post,
                    function=self.function_to_learn
                )

            # -- probes
            # self.pre_probe = nengo.Probe(self.model.pre, synapse=0.01)
            self.post_probe = nengo.Probe(self.model.post, synapse=0.01)
            self.ground_truth_probe = nengo.Probe(self.model.ground_truth, synapse=0.01)
            # self.weights_probe = nengo.Probe(self.model.conn, 'weights', synapse=None)
            # self.model.error_probe = nengo.Probe(self.model.error, synapse=0.01)
        self.sim = nengo.Simulator(self.model, progress_bar=verbose)

        # hacky way to get the mPES operator and the initial memristor weights from it
        mpes_op = get_operator_from_sim(self.sim, 'SimmPES')
        if mpes_op is not None:
            # save initial weights to then use with PES
            with open(f'{tmp_folder}/initial_weights_{self.seed}.npy', 'wb') as f:
                np.save(f, self.sim.signals[mpes_op.weights], allow_pickle=False)

        self.sim.run(self.sim_time)

        self.results_ready = True

        return self

    def predict(self, X):
        assert self.results_ready, "You must call fit() before calling predict()"

        return np.array([0])

    def score(self, X, y=None, sample_weight=None):
        assert self.results_ready, "You must call fit() before calling score()"

        ground_truth_data = np.array_split(self.sim.data[self.ground_truth_probe], self.num_blocks)
        post_data = np.array_split(self.sim.data[self.post_probe], self.num_blocks)
        # extract learning blocks
        # train_ground_truth_data = np.array([x for i, x in enumerate(ground_truth_data) if i % 2 != 0])
        test_ground_truth_data = np.array([x for i, x in enumerate(ground_truth_data) if i % 2 == 0])
        # extract testing blocks
        # train_post_data = np.array([x for i, x in enumerate(post_data) if i % 2 != 0])
        test_post_data = np.array([x for i, x in enumerate(post_data) if i % 2 == 0])

        # compute testing error for learn network
        testing_errors = np.sum(np.sum(np.abs(test_post_data - test_ground_truth_data), axis=1), axis=1)[1:]

        if isinstance(self.learning_rule, mPES):
            lr = 'mpes'
        elif isinstance(self.learning_rule, PES):
            lr = 'pes'
        else:
            lr = 'nef'

        with open(f"{tmp_folder}/testing_errors_{lr}_{self.seed}.csv", "w") as f:
            np.savetxt(f, testing_errors, delimiter=",")

        if self.low_memory:
            del self.sim

        return np.mean(testing_errors)

    # TODO finish plot function for single iteration
    def plot(self):
        assert self.results_ready, "You must call fit() before calling plot()"
        assert not self.low_memory, "You must set low_memory=True before calling plot()"

        size_L = 10
        size_M = 8
        size_S = 6
        fig, ax = plt.subplots(figsize=(12, 10), dpi=72)
        # fig.set_size_inches((3.5, 3.5 * ((5. ** 0.5 - 1.0) / 2.0)))
        fig.suptitle(self.exp_name, fontsize=size_L)
        ax.set_ylabel("Total error", fontsize=size_M)
        ax.set_xlabel("Seconds", fontsize=size_M)
        ax.tick_params(axis='x', labelsize=size_S)
        ax.tick_params(axis='y', labelsize=size_S)

        x = (np.arange(self.num_testing_blocks) * 2 * self.learn_block_time) + self.learn_block_time

        ax.legend(loc="best", fontsize=size_S)
        fig.tight_layout()

        return fig

    def count_pulses(self):
        assert self.results_ready, "You must call fit() before calling count_pulses()"

        mpes_op = get_operator_from_sim(self.sim, 'SimmPES')

        # -- evaluate the average length of consecutive reset or set pulses
        consec_pos_set, consec_pos_reset = average_number_consecutive_pulses(mpes_op.pos_pulse_archive)
        consec_neg_set, consec_neg_reset = average_number_consecutive_pulses(mpes_op.neg_pulse_archive)
        print('Average length of consecutive SET pulses')
        print(np.mean([consec_pos_set, consec_neg_set]))
        print('Average length of consecutive RESET pulses')
        print(np.mean([consec_pos_reset, consec_neg_reset]))

        self.num_pos_set, self.num_pos_reset = average_number_pulses(mpes_op.pos_pulse_archive)
        self.num_neg_set, self.num_neg_reset = average_number_pulses(mpes_op.neg_pulse_archive)
        print('Average number of SET pulses')
        print(np.mean([self.num_pos_set, self.num_neg_set]))
        print('Average number of RESET pulses')
        print(np.mean([self.num_pos_reset, self.num_neg_reset]))

    def energy_consumption(self):
        assert self.results_ready, "You must call fit() before calling power_consumption()"

        mpes_op = get_operator_from_sim(self.sim, 'SimmPES')

        self.mean_energy = np.mean((mpes_op.energy_pos, mpes_op.energy_neg))
        print(f'Average energy consumption {order_of_magnitude.prefix(self.mean_energy)[2]}J')

        if not hasattr(self, 'num_pos_set ') and \
                not hasattr(self, 'num_pos_reset') and \
                not hasattr(self, 'num_neg_set') and \
                not hasattr(self, 'num_neg_reset'):
            self.count_pulses()
        self.energy_per_pulse = self.mean_energy / (
                self.num_pos_set + self.num_pos_reset + self.num_neg_set + self.num_neg_reset)
        print(f'Average energy consumption per pulse {order_of_magnitude.prefix(self.energy_per_pulse)[2]}J')


while True:
    experiment = input('Choose an experiment (1-5): ')
    if experiment in ['1', '2', '3', '4', '5']:
        break
    else:
        print('Invalid input. Please try again.')
experiment = int(experiment)

experiment_dict = {
    1: "Multiplying two numbers",
    2: "Combining two products",
    3: "Three separate products",
    4: "Two-dimensional circular convolution",
    5: "Three-dimensional circular convolution"
}

iterations = 10
num_cpus = -1
print(f"Experiment: {experiment_dict[experiment]} ({experiment})")

seed = np.random.randint(0, 2 ** 32 - 1)
dummy_param_grid = {
    'seed': [s for s in range(seed, seed + iterations)]
}
print(f"Param grid/seeds: {dummy_param_grid}")

while True:
    estimate_time = input('Estimate time? (y/n): ')
    if estimate_time == 'y':
        estimate_search_time(Trevor_Estimator(experiment=experiment, learning_rule=mPES(low_memory=True)),
                             dummy_param_grid,
                             num_cpus=num_cpus, cv=1, repeat=iterations)
    elif estimate_time == 'n':
        break
    else:
        print('Invalid input. Please try again.')

"""[(slice(None), slice(None))] is a hack to make GridSearchCV use only one CV fold"""
gs_mpes = GridSearchCV(
    Trevor_Estimator(experiment=experiment, learning_rule=mPES(low_memory=True), low_memory=True),
    param_grid=dummy_param_grid,
    n_jobs=num_cpus, verbose=2, cv=[(slice(None), slice(None))])
gs_mpes.fit([0])
gs_pes = GridSearchCV(
    Trevor_Estimator(experiment=experiment, learning_rule=PES(), low_memory=True),
    param_grid=dummy_param_grid,
    n_jobs=num_cpus, verbose=2, cv=[(slice(None), slice(None))])
gs_pes.fit([0])
gs_nef = GridSearchCV(
    Trevor_Estimator(experiment=experiment, learning_rule=None, low_memory=True),
    param_grid=dummy_param_grid,
    n_jobs=num_cpus, verbose=2, cv=[(slice(None), slice(None))])
gs_nef.fit([0])

# read errors from files
errors = {}
num_testing_blocks = gs_mpes.estimator.num_testing_blocks
learn_block_time = gs_mpes.estimator.learn_block_time
for f in glob.glob(f'{tmp_folder}/testing_errors_*.csv'):
    lr = re.search(f'{tmp_folder}/testing_errors_(.*)_(.*).csv', f).group(1)
    if lr not in errors:
        errors[lr] = np.empty((0, num_testing_blocks))
    errors[lr] = np.vstack((errors[lr], np.genfromtxt(f, delimiter=',')))

pickle.dump(errors, open(f'{tmp_folder}/testing_errors_exp_{experiment}.pkl', 'wb'))

# compute mean testing error and confidence intervals
ci_mpes = ci(errors['mpes'])
ci_pes = ci(errors['pes'])
ci_nef = ci(errors['nef'])

# plot testing error
size_L = 10
size_M = 8
size_S = 6
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
# fig.set_size_inches((3.5, 3.5 * ((5. ** 0.5 - 1.0) / 2.0)))
fig.suptitle(experiment_dict[experiment], fontsize=size_L)
ax.set_ylabel("Total error", fontsize=size_M)
ax.set_xlabel("Seconds", fontsize=size_M)
ax.tick_params(axis='x', labelsize=size_S)
ax.tick_params(axis='y', labelsize=size_S)

x = (np.arange(num_testing_blocks) * 2 * learn_block_time) + learn_block_time

ax.plot(x, ci_mpes[0], label="Learned (mPES)", c="g")
ax.plot(x, ci_mpes[1], linestyle="--", alpha=0.5, c="g")
ax.plot(x, ci_mpes[2], linestyle="--", alpha=0.5, c="g")
ax.fill_between(x, ci_mpes[1], ci_mpes[2], alpha=0.3, color="g")
ax.plot(x, ci_pes[0], label="Control (PES)", c="b")
ax.plot(x, ci_pes[1], linestyle="--", alpha=0.5, c="b")
ax.plot(x, ci_pes[2], linestyle="--", alpha=0.5, c="b")
ax.fill_between(x, ci_pes[1], ci_pes[2], alpha=0.3, color="b")
ax.plot(x, ci_nef[0], label="Control (NEF)", c="r")
ax.plot(x, ci_nef[1], linestyle="--", alpha=0.5, c="r")
ax.plot(x, ci_nef[2], linestyle="--", alpha=0.5, c="r")
ax.fill_between(x, ci_nef[1], ci_nef[2], alpha=0.5, color="r")
ax.fill_between(x, ci_mpes[1], ci_mpes[2], alpha=0.3, color="g")

# -- plot linear regression too
txt_anchor = ''
for y, col, lr in zip([ci_mpes[0], ci_pes[0], ci_nef[0]], ['g', 'b', 'r'], ['mPES', 'PES', 'NEF']):
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x, poly1d_fn(x), f'--{col}')
    print(f'{lr} learning linear regression slope: {np.polyder(poly1d_fn)[0]}')
    txt_anchor += f'{lr} slope: {np.polyder(poly1d_fn)[0]:.2f}\n'

anchored_text = AnchoredText(txt_anchor[:-1], loc=2)
ax.add_artist(anchored_text)

ax.legend(loc="best", fontsize=size_M)
fig.tight_layout()
fig.show()
fig.savefig(f'{tmp_folder}/testing_errors_exp_{experiment}.png', dpi=300)

while True:
    save = input('Save results? (y/n)')
    if save == 'y':
        os.rename(f'{tmp_folder}/testing_errors_exp_{experiment}.pkl', f'./testing_errors_exp_{experiment}_SAVED.pkl')
        os.rename(f'{tmp_folder}/testing_errors_exp_{experiment}.png', f'./testing_errors_exp_{experiment}_SAVED.png')
        cleanup()
        exit()
    elif save == 'n':
        cleanup()
        exit()
    else:
        print("Please enter 'y' or 'n'")
