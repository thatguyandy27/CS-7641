import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import numpy as np

import experiments


class BenchmarkExperiment(experiments.BaseExperiment):

    def __init__(self, details, verbose=False):
        super().__init__(details)
        d = 10
        self._verbose = verbose
        self._nn_arch = [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
        self._nn_reg = [10 ** -x for x in range(1, 5)]
        self._clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        # self.learning_rates = [0.004, 0.008, 0.016]
        self.activation = ['relu', 'logistic']


    def experiment_name(self):
        return 'benchmark'

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/benchmark.py
        self.log("Performing {}".format(self.experiment_name()))

        # %% benchmarking for chart type 2
        grid = {'NN__alpha': self._nn_reg,
                'NN__activation': self.activation,
                # 'NN__learning_rate_init': self.learning_rates,
                'NN__hidden_layer_sizes': self._nn_arch
                }
        mlp = MLPClassifier()
        pipe = Pipeline([('NN', mlp)], memory=experiments.pipeline_memory)
        gs, _ = self.gs_with_best_estimator(pipe, grid)
        self.log("Grid search complete")

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_nn_bmk.csv'.format(self._details.ds_name)))
        self.log("Done")

        # benchmark based on best params from assignment 1
        mlp = MLPClassifier()
        pipe = Pipeline([('NN', mlp)], memory=experiments.pipeline_memory)
        gs, _ = self.gs_with_best_estimator(pipe, experiments.BEST_NN_PARAMS[self._details.ds_name], type='ass1')

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_ass1_nn_bmk.csv'.format(self._details.ds_name)))

        # Run clustering as a subexperiment
        self.log("Running clustering sub-experiment")
        experiments.run_subexperiment(self, self._out.format('clustering/'))

    def perform_cluster(self, dim_param):
        self.log('Clustering for a specific dim is not run for {}'.format(self.experiment_name()))
