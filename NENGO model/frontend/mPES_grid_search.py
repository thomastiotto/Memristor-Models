from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
import pandas as pd

from sklearn.metrics import mean_squared_error
from yakopcic_learning_new import mPES

from extras import *


class mPES_Estimator(BaseEstimator, RegressorMixin):
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __init__(self, gain=1e5, voltages=[-1.6300607380628072, 0.006566045887405729], probability=0.1,
                 low_memory=False):
        nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'
        nengo.rc['decoder_cache']['enabled'] = 'False'

        self.low_memory = low_memory

        self.results_ready = False

        self.gain = gain
        self.voltages = voltages
        self.probability = probability

        self.timestep = 0.001
        dimensions = 3
        pre_n_neurons = post_n_neurons = error_n_neurons = 100
        sim_time = 30
        self.noise_percent = 0.15
        input_function_train = input_function_test = Sines(period=4)
        self.learn_time = int(sim_time * 3 / 4)
        self.test_time = sim_time - self.learn_time
        self.function_to_learn = lambda x: x
        self.strategy = 'symmetric-probabilistic'

        self.model = nengo.Network()
        with self.model:
            nengo_dl.configure_settings(inference_only=True)
            # Create an input node
            input_node = nengo.Node(
                output=SwitchInputs(input_function_train,
                                    input_function_test,
                                    switch_time=self.learn_time),
                size_out=dimensions
            )

            # Shut off learning by inhibiting the error population
            stop_learning = nengo.Node(output=lambda t: t >= self.learn_time)

            # Create the ensemble to represent the input, the learned output, and the error
            pre = nengo.Ensemble(pre_n_neurons, dimensions=dimensions)
            post = nengo.Ensemble(post_n_neurons, dimensions=dimensions)
            self.model.error = nengo.Ensemble(error_n_neurons, dimensions=dimensions, radius=2)

            # Connect pre and post with a communication channel
            # the matrix given to transform is the initial weights found in model.sig[conn]["weights"]
            # the initial transform has no influence on learning because it is overwritten by mPES
            # the only influence is on the very first timesteps, before the error becomes large enough
            self.model.conn = nengo.Connection(
                pre.neurons,
                post.neurons,
                transform=np.zeros((post.n_neurons, pre.n_neurons))
            )

            # Compute the error signal (error = actual - target)
            nengo.Connection(post, self.model.error)

            # Subtract the target (this would normally come from some external system)
            nengo.Connection(pre, self.model.error, function=self.function_to_learn, transform=-1)

            # Connect the input node to ensemble pre
            nengo.Connection(input_node, pre)

            nengo.Connection(
                stop_learning,
                self.model.error.neurons,
                transform=-20 * np.ones((self.model.error.n_neurons, 1)))

            # essential ones are used to calculate the statistics
            self.pre_probe = nengo.Probe(pre, synapse=0.01)
            self.post_probe = nengo.Probe(post, synapse=0.01)

            # optional ones are used to plot the results
            # self.input_node_probe = nengo.Probe(input_node)
            # self.error_probe = nengo.Probe(self.model.error, synapse=0.01)
            # self.learn_probe = nengo.Probe(stop_learning, synapse=None)
            # self.post_spikes_probe = nengo.Probe(post.neurons)

    def fit(self, X, y=None):
        with self.model:
            # Apply the learning rule to conn
            self.model.conn.learning_rule_type = mPES(noisy=self.noise_percent, gain=self.gain,
                                                      strategy=self.strategy,
                                                      resetV=self.voltages[0], setV=self.voltages[1],
                                                      resetP=self.probability, setP=self.probability)
            # Provide an error signal to the learning rule
            nengo.Connection(self.model.error, self.model.conn.learning_rule)

            # learning rule probes
            # self.x_pos_probe = nengo.Probe(self.model.conn.learning_rule, "x_pos", synapse=None)
            # self.x_neg_probe = nengo.Probe(self.model.conn.learning_rule, "x_neg", synapse=None)
            # self.weight_probe = nengo.Probe(self.model.conn, "weights", synapse=None)

        self.sim = nengo.Simulator(self.model)
        self.sim.run(self.learn_time)
        self.sim.run(self.test_time)

        self.results_ready = True

        return self

    def predict(self, X):
        self.sim.run(self.test_time)

        learning_time = int((self.learn_time / self.timestep))
        y = self.sim.data[self.post_probe][learning_time:, ...]

        return y

    def score(self, X, y=None, sample_weight=None):
        assert self.results_ready, "You must call fit() before calling score()"

        y_true = self.sim.data[self.pre_probe][int((self.learn_time / self.timestep)):, ...]
        y_pred = self.sim.data[self.post_probe][int((self.learn_time / self.timestep)):, ...]

        # MSE after learning
        mse = mean_squared_error(self.function_to_learn(y_true), y_pred, multioutput='raw_values')
        # Correlation coefficients after learning
        correlation_coefficients = correlations(self.function_to_learn(y_true), y_pred)

        if self.low_memory:
            # del self.pre_probe
            # del self.post_probe
            del self.sim

        return np.mean(mse_to_rho_ratio(mse, correlation_coefficients[1]))

    def plot(self, smooth=False):
        assert self.results_ready, "You must call fit() before calling plot()"
        assert not self.low_memory, "You must set low_memory=True before calling plot()"

        pre = self.function_to_learn(self.sim.data[self.pre_probe])
        post = self.sim.data[self.post_probe]

        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=False)
        fig.set_size_inches((12, 8))

        learning_time = int((self.learn_time / self.timestep))
        time = self.sim.trange()[learning_time:, ...]
        pre = pre[learning_time:, ...]
        post = post[learning_time:, ...]

        axes[0, 0].xaxis.set_tick_params(labelsize='xx-large')
        axes[0, 0].yaxis.set_tick_params(labelsize='xx-large')
        # axes[0, 0].set_ylim(-1, 1)

        if smooth:
            from scipy.signal import savgol_filter

            pre = np.apply_along_axis(savgol_filter, 0, pre, window_length=51, polyorder=3)
            post = np.apply_along_axis(savgol_filter, 0, post, window_length=51, polyorder=3)

        axes[0, 0].plot(
            time,
            pre,
            # linestyle=":",
            alpha=0.3,
            label='Pre')
        axes[0, 0].set_prop_cycle(None)
        axes[0, 0].plot(
            time,
            post,
            label='Post')
        # if self.n_dims <= 3:
        #     axes[ 0, 0 ].legend(
        #             [ f"Pre dim {i}" for i in range( self.n_dims ) ] +
        #             [ f"Post dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        # axes[ 0, 0 ].set_title( "Pre and post decoded on testing phase", fontsize=16 )

        plt.tight_layout()

        return fig

    def count_pulses(self):
        assert self.results_ready, "You must call fit() before calling count_pulses()"

        mpes_op = get_operator_from_sim(self.sim, 'SimmPES')

        # -- evaluate number of memristor pulses over simulation
        pos_pulse_counter = mpes_op.set_pulse_counter
        neg_pulse_counter = mpes_op.reset_pulse_counter
        print('Average number of SET pulses')
        print(np.mean(pos_pulse_counter))
        print('Average number of RESET pulses')
        print(np.mean(neg_pulse_counter))

        # -- evaluate the average length of consecutive reset or set pulses
        consec_pos_set, consec_pos_reset = average_number_consecutive_pulses(mpes_op.pos_pulse_archive)
        consec_neg_set, consec_neg_reset = average_number_consecutive_pulses(mpes_op.neg_pulse_archive)
        print('Average length of consecutive SET pulses')
        print(np.mean([consec_pos_set, consec_neg_set]))
        print('Average length of consecutive RESET pulses')
        print(np.mean([consec_pos_reset, consec_neg_reset]))

        num_pos_set, num_pos_reset = average_number_pulses(mpes_op.pos_pulse_archive)
        num_neg_set, num_neg_reset = average_number_pulses(mpes_op.neg_pulse_archive)
        print('Average number of SET pulses')
        print(np.mean([num_pos_set, num_neg_set]))
        print('Average number of RESET pulses')
        print(np.mean([num_pos_reset, num_neg_reset]))


num_par = 10
# -- define grid search parameters
param_grid = {
    'gain': np.logspace(np.rint(4).astype(int), np.rint(6).astype(int),
                        num=np.rint(num_par).astype(int)),
    # 'gain': [1e3, 1e6],
    'probability': np.logspace(np.rint(-2).astype(int), np.rint(0).astype(int),
                               num=np.rint(num_par).astype(int)),
    # 'probability': [0.1, 0.2, 0.3, 0.4, 0.5],
    'voltages': [[-2.1331527635533685, 0.011873603203071863],
                 [-1.6300607380628072, 0.006566045887405729],
                 [-1.3180811362586602, 0.004382346688619062]]
}
param_grid_fast = {
    'gain': [1e3, 1e6]
}

gs = GridSearchCV(mPES_Estimator(low_memory=True), param_grid=param_grid_fast, cv=2,
                  n_jobs=-1, verbose=3)
gs.fit(np.zeros(30000))
pd_results = pd.DataFrame(gs.cv_results_)
print(gs.best_params_)
print(gs.best_score_)

estimator = mPES_Estimator(**gs.best_params_)
estimator.fit([0])
print(estimator.score([0]))
estimator.plot().show()

# gsh = HalvingGridSearchCV(mPES_Estimator(), param_grid=param_grid_fast, cv=2, return_train_score=False,
#                           n_jobs=-1, verbose=3)
# gsh.fit(np.zeros(30000))
# pd_results_h = pd.DataFrame(gsh.cv_results_)
# print(gsh.best_params_)
# print(gsh.best_score_)
