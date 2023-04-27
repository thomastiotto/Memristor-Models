# -*- coding: utf-8 -*-
"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Bernoulli Institute
"""

import atexit
import copy
import glob
import itertools
import os
import shutil
import signal
import warnings
from datetime import datetime

import nengo
import numpy as np
import pandas as pd
import pathos
import tqdm_pathos
from dill import PicklingWarning, UnpicklingWarning
from matplotlib import pyplot as plt
from nengo.processes import WhiteSignal
from numpy import VisibleDeprecationWarning
from tqdm import TqdmWarning

from yakopcic_learning_new import mPES

warnings.simplefilter("ignore", PicklingWarning)
warnings.simplefilter("ignore", UnpicklingWarning)
warnings.simplefilter("ignore", VisibleDeprecationWarning)


def cleanup(exit_code=None, frame=None):
    try:
        print("Cleaning up leftover files...")
        # delete tmp folder
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        pass


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def product_dict_to_list(**kwargs):
    from itertools import product

    vals = kwargs.values()

    out_list = []
    for instance in product(*vals):
        out_list.append(list(instance))

    return out_list


def run_sim(params, plot=False):
    warnings.filterwarnings("ignore", category=TqdmWarning)

    if isinstance(params, list):
        gain = params[0]
        resetP = params[1]
        setP = params[2]
        resetV = params[3]
        setV = params[4]
    elif isinstance(params, dict) or isinstance(params, pd.Series):
        gain = params['gain'] if 'gain' in params else 2153
        resetP = params['resetP'] if 'resetP' in params else 1
        setP = params['setP'] if 'setP' in params else 0.1
        resetV = params['resetV'] if 'resetV' in params else -1
        setV = params['setV'] if 'setV' in params else 0.25
    worker_id = params[5] if len(params) > 5 else None

    readV = -0.01
    noise = 0.15

    pid = os.getpid()
    # print(f'RUNNING worker {pid} with params: {params}')

    nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'
    nengo.rc['decoder_cache']['enabled'] = 'False'

    experiment = 5

    if experiment == 1:
        exp_string = "PRODUCT experiment"
        exp_name = "Multiplying two numbers"
        function_to_learn = lambda x: x[0] * x[1]
        # [ pre, post, ground_truth, error ]
        neurons = [200, 200, 100, 100]
        dimensions = [2, 1, 1, 1]
        sim_time = 50
        img_name = 'product'
    if experiment == 2:
        exp_string = "COMBINED PRODUCTS experiment"
        exp_name = "Combining two products"
        function_to_learn = lambda x: x[0] * x[1] + x[2] * x[3]
        # [ pre, post, ground_truth, error ]
        neurons = [400, 400, 100, 100]
        dimensions = [4, 1, 1, 1]
        sim_time = 100
        img_name = 'combined_products'
    if experiment == 3:
        exp_string = "SEPARATE PRODUCTS experiment"
        exp_name = "Three separate products"
        function_to_learn = lambda x: [x[0] * x[1], x[0] * x[2], x[1] * x[2]]
        # [ pre, post, ground_truth, error ]
        neurons = [300, 300, 300, 300]
        dimensions = [3, 3, 3, 3]
        sim_time = 100
        img_name = 'separate_products'
    if experiment == 4:
        exp_string = "2D CIRCULAR CONVOLUTIONS experiment"
        exp_name = "Two-dimensional circular convolution"
        # [ pre, post, ground_truth, error, conv ]
        neurons = [400, 400, 200, 200, 200]
        dimensions = [4, 2, 2, 2, 2]
        function_to_learn = lambda x: np.fft.ifft(
            np.fft.fft(x[:int(dimensions[0] / 2)]) * np.fft.fft(x[int(dimensions[0] / 2):])
        )
        sim_time = 200
        img_name = '2d_cconv'
    if experiment == 5:
        exp_string = "3D CIRCULAR CONVOLUTIONS experiment"
        exp_name = "Three-dimensional circular convolution"
        # [ pre, post, ground_truth, error, conv ]
        neurons = [600, 300, 300, 300, 300]
        dimensions = [6, 3, 3, 3, 3]
        function_to_learn = lambda x: np.fft.ifft(
            np.fft.fft(x[:int(dimensions[0] / 2)]) * np.fft.fft(x[int(dimensions[0] / 2):])
        )
        # TODO changed from 400 to 200
        sim_time = 200
        img_name = '3d_cconv'

    learn_block_time = 2.5
    # to have an extra testing block at t=[0,2.5]
    sim_time += learn_block_time
    convolve = False if experiment <= 3 else True

    num_blocks = int(sim_time / learn_block_time)
    num_testing_blocks = int(num_blocks / 2)

    model = nengo.Network()
    with model:
        model.inp = nengo.Node(
            # WhiteNoise( dist=Gaussian( 0, 0.05 ), seed=seed ),
            WhiteSignal(sim_time, high=5),
            size_out=dimensions[0]
        )
        model.pre = nengo.Ensemble(neurons[0], dimensions=dimensions[0])
        model.post = nengo.Ensemble(neurons[1], dimensions=dimensions[1])
        model.ground_truth = nengo.Ensemble(neurons[2], dimensions=dimensions[2])

        nengo.Connection(model.inp, model.pre)

        if convolve:
            model.conv = nengo.networks.CircularConvolution(neurons[4], dimensions[4])
            nengo.Connection(model.inp[:int(dimensions[0] / 2)],
                             model.conv.input_a,
                             synapse=None)
            nengo.Connection(model.inp[int(dimensions[0] / 2):],
                             model.conv.input_b,
                             synapse=None)
            nengo.Connection(model.conv.output, model.ground_truth,
                             synapse=None)
        else:
            nengo.Connection(model.inp, model.ground_truth,
                             function=function_to_learn,
                             synapse=None)

        model.error = nengo.Ensemble(neurons[3], dimensions=dimensions[3])

        model.conn = nengo.Connection(
            model.pre.neurons,
            model.post.neurons,
            transform=np.zeros((model.post.n_neurons, model.pre.n_neurons)),
            learning_rule_type=mPES(low_memory=True,
                                    gain=gain,
                                    noise_percentage=noise,
                                    resetP=resetP,
                                    setP=setP,
                                    resetV=resetV,
                                    setV=setV,
                                    readV=readV,
                                    strategy='symmetric-probabilistic',
                                    read_enabled=True,
                                    verbose=False),
        )

        nengo.Connection(model.error, model.conn.learning_rule)
        nengo.Connection(model.post, model.error)
        nengo.Connection(model.ground_truth, model.error, transform=-1)

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

        model.inhib = nengo.Node(cyclic_inhibit(learn_block_time).step)
        nengo.Connection(model.inhib, model.error.neurons,
                         transform=[[-1]] * model.error.n_neurons)

        # -- probes
        # pre_probe = nengo.Probe(model.pre, synapse=0.01)
        post_probe = nengo.Probe(model.post, synapse=0.01)
        ground_truth_probe = nengo.Probe(model.ground_truth, synapse=0.01)
        # weights_probe = nengo.Probe(model.conn, 'weights', synapse=None)
        # error_probe = nengo.Probe(model.error, synapse=0.01)

    sim = nengo.Simulator(model, progress_bar=False)

    sim.run(sim_time)

    ground_truth_data = np.array_split(sim.data[ground_truth_probe], num_blocks)
    post_data = np.array_split(sim.data[post_probe], num_blocks)
    # extract learning blocks
    # train_ground_truth_data = np.array([x for i, x in enumerate(ground_truth_data) if i % 2 != 0])
    test_ground_truth_data = np.array([x for i, x in enumerate(ground_truth_data) if i % 2 == 0])
    # extract testing blocks
    # train_post_data = np.array([x for i, x in enumerate(post_data) if i % 2 != 0])
    test_post_data = np.array([x for i, x in enumerate(post_data) if i % 2 == 0])

    # compute testing error for learn network
    testing_errors = np.sum(np.sum(np.abs(test_post_data - test_ground_truth_data), axis=1), axis=1)[1:]

    lr = 'mpes'

    # with open(f"{tmp_folder}/testing_errors_{lr}.csv", "w") as f:
    #     np.savetxt(f, testing_errors, delimiter=",")

    score = np.mean(testing_errors)

    size_L = 10
    size_M = 8
    size_S = 6
    fig, ax = plt.subplots(figsize=(12, 10), dpi=72)
    # fig.set_size_inches((3.5, 3.5 * ((5. ** 0.5 - 1.0) / 2.0)))
    fig.suptitle(exp_name, fontsize=size_L)
    ax.set_ylabel("Total error", fontsize=size_M)
    ax.set_xlabel("Seconds", fontsize=size_M)
    ax.tick_params(axis='x', labelsize=size_S)
    ax.tick_params(axis='y', labelsize=size_S)

    x = (np.arange(num_testing_blocks) * 2 * learn_block_time) + learn_block_time
    ax.plot(x, testing_errors, label=f"{lr}", linewidth=1.0)

    ax.legend(loc="best", fontsize=size_S)
    fig.tight_layout()
    fig.savefig(f"{tmp_folder}/error_{lr}_{pid}.png", dpi=72)
    if plot:
        fig.show()

    # cleanup
    # shutil.rmtree(tmp_folder)
    del sim, model

    return params, score


if __name__ == '__main__':
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(tmp_folder)

    num_cpus = pathos.multiprocessing.cpu_count()
    num_par = 2
    cv = 1
    param_grid = {
        'gain': np.logspace(3, 6, num=num_par),
        'resetV': np.linspace(-4, -0.25, num=num_par),
        'setV': np.linspace(0.1, 1, num=num_par),
        'resetP': np.linspace(0.1, 1, num=num_par),
        'setP': np.linspace(0.1, 1, num=num_par),
    }
    gain = 2153
    resetV = -1
    setV = 0.25
    resetP = 1
    setP = 0.1
    param_grid = {
        'gain': [gain / 2, gain, gain * 2],
        'resetV': [resetV / 2, resetV, resetV * 2],
        'setV': [setV / 2, setV, setV * 2],
        'resetP': [resetP / 4, resetP / 2, resetP],
        'setP': [setP / 2, setP, setP * 2],
    }
    sweeped_params = [(k, i) for i, (k, v) in enumerate(param_grid.items()) if len(v) > 1]

    print(param_grid)

    param_grid_pool = list(
        itertools.chain.from_iterable(map(copy.copy, product_dict_to_list(**param_grid)) for _ in range(cv)))
    # hack to keep track of process ids
    for i, g in enumerate(param_grid_pool):
        g.append(i % num_cpus)

    chunked_param_grid = list(chunks(param_grid_pool, num_cpus))

    # with pathos.multiprocessing.ProcessPool(num_cpus) as p:
    #     results = p.map(partial(run_sim, plot=False), param_grid_pool)
    results = tqdm_pathos.map(run_sim, param_grid_pool, n_cpus=num_cpus)

    if len(sweeped_params) >= 1:
        res_unsweeped_removed = []
        for r in results:
            res_tmp = []
            for p in sweeped_params:
                res_tmp.append(r[0][p[1]])
            res_unsweeped_removed.append((res_tmp, r[1]))
    else:
        res_unsweeped_removed = results

    sweeped_param_names = [p[0] for p in sweeped_params]

    # create a dataframe with the raw results
    df_results = pd.DataFrame(columns=sweeped_param_names + ['score'])
    for r in res_unsweeped_removed:
        df_results.loc[len(df_results)] = r[0] + [r[1]]

    # aggregate results by the sweeped parameters
    df_results_aggregated = df_results.copy()
    df_results_aggregated = df_results_aggregated.groupby(sweeped_param_names).mean().reset_index()
    df_results_aggregated.sort_values(by=sweeped_param_names, ascending=True, inplace=True)
    df_results_aggregated.to_csv(f'{tmp_folder}/results.csv')

    # print best results in order
    print(df_results_aggregated[sweeped_param_names + ['score']].sort_values(by='score', ascending=False))

    # plot results for cue time
    # fig, ax = plt.subplots(figsize=(12, 10))
    # df_results_aggregated.plot(kind='bar', ax=ax, x='cue_time', y='score', legend=False)
    # ax.set_ylabel('Score')
    # ax.set_xlabel(sweeped_param_names)
    # # TODO the labels look wrong still
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.set_title('Score for different parameters')
    # fig.savefig(f'{tmp_folder}/score.png')
    # fig.show()

    # Run model with the best parameters and plot output
    best_params = df_results_aggregated.loc[df_results_aggregated.score.idxmax()]
    best_score = best_params['score']
    best_params = best_params.drop('score').to_dict()
    print(f'Best parameters: {best_params}')
    run_sim(best_params)

    while True:
        save = input('Save results? (y/n)')
        if save == 'y':
            save_folder = f'SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
            os.makedirs(save_folder)
            # os.rename(f'{tmp_folder}/score.png', f'{save_folder}/score.png')
            os.rename(f'{tmp_folder}/results.csv', f'{save_folder}/results.csv')

            save2 = input('Saved overall results.  Also save individual plots?: (y/n)')
            if save2 == 'y':
                for f in glob.glob(f'{tmp_folder}/**/*.png', recursive=True):
                    pid = os.path.normpath(f).split(os.path.sep)[1]
                    os.makedirs(f'{save_folder}/{pid}', exist_ok=True)
                    os.rename(f, f'{save_folder}/{pid}/{os.path.split(f)[1]}')

            cleanup()
            break
        elif save == 'n':
            cleanup()
            break
        else:
            print("Please enter 'y' or 'n'")
