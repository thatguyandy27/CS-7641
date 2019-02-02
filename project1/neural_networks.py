from sklearn import neural_network
import exp
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def train(x, y, hidden_layer_sizes=(20, 5), activation='relu', alpha=0.0001, batch_size='auto',
             learning_rate='constant', learning_rate_init=0.001, power_t=.5, max_iter=200,
             shuffle=True, random_state=None, tol=1e-4, ):
    clf = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha,
                                       batch_size=batch_size, learning_rate=learning_rate,
                                       learning_rate_init=learning_rate_init, power_t=power_t,
                                       max_iter=max_iter, shuffle=shuffle, random_state=random_state,
                                       tol=tol)
    clf = clf.fit(x, y)

    return clf


def run_experiment(x, y, x_train, y_train, x_test, y_test, ds_name, class_list, max_depth=None):
    alphas = [10 ** -x for x in np.arange(-1, 9.01, 1)]

    # TODO: Allow for better tuning of hidden layers based on dataset provided
    d = min(x.shape[1], 5)
    hiddens = [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
    learning_rates = sorted([(2 ** x) / 1000 for x in range(5)] + [0.000001])

    params = {'CLF__activation': ['relu', 'logistic'], 'CLF__alpha': alphas,
              'CLF__learning_rate_init': learning_rates,
              'CLF__hidden_layer_sizes': hiddens}

    clf = neural_network.MLPClassifier()

    pipe = Pipeline([('Scale', StandardScaler()),
                     ('CLF', clf)])

    cv = exp.basic_results(pipe, 'NeuralNet', ds_name, x_train, y_train, x_test, y_test, class_list, params)

    exp.make_complexity_curve(cv.best_estimator_, 'NeuralNet', ds_name, x, y, 'CLF__alpha', 'Alpha', alphas, 'log')

