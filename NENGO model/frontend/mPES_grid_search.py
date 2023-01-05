from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from nengo.learning_rules import PES
from nengo.params import Default
from nengo.processes import WhiteSignal
from sklearn.metrics import mean_squared_error
from yakopcic_learning_new import mPES
from tqdm import tqdm

from extras import *


class mPES_Estimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'

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

    def fit(self, **kwargs):
        with self.model:
            # Apply the learning rule to conn
            self.model.conn.learning_rule_type = mPES(noisy=self.noise_percent, gain=kwargs['gain'],
                                                      strategy=self.strategy)
            # Provide an error signal to the learning rule
            nengo.Connection(self.model.error, self.model.conn.learning_rule)

        self.sim = nengo.Simulator(self.model)
        self.sim.run(self.learn_time)

    def predict(self):
        self.sim.run(self.test_time)

        # -- evaluate number of memristor pulses over simulation
        mpes_op = get_operator_from_sim(self.sim, 'SimmPES')

        # essential statistics
        y_true = self.sim.data[self.pre_probe][int((self.learn_time / self.timestep)):, ...]
        y_pred = self.sim.data[self.post_probe][int((self.learn_time / self.timestep)):, ...]
        # MSE after learning
        mse = mean_squared_error(self.function_to_learn(y_true), y_pred, multioutput='raw_values')
        # Correlation coefficients after learning
        correlation_coefficients = correlations(self.function_to_learn(y_true), y_pred)

        return mse_to_rho_ratio(mse, correlation_coefficients[1])


estimator = mPES_Estimator()
estimator.fit(gain=1e5)
print(estimator.predict())

voltage_options = [[-2.1331527635533685, 0.011873603203071863],
                   [-1.6300607380628072, 0.006566045887405729],
                   [-1.3180811362586602, 0.004382346688619062]]

# -- define grid search parameters
param_grid = {
    'gain': [1e3, 1e4, 1e5, 1e6],
    'probability': [0.1, 0.2, 0.3, 0.4, 0.5],
    'voltages': voltage_options
}
