from sklearn import ensemble
import exp
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(x, y, base_estimator=None, n_estimators=50, learning_rate=1., algorithm='SAMME.R',
            random_state=None):
    clf = ensemble.AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators,
                                      learning_rate=learning_rate, algorithm=algorithm,
                                      random_state=random_state)
    clf = clf.fit(x, y)

    return clf



def run_experiment(x, y, x_train, y_train, x_test, y_test, ds_name, class_list, max_depth=None):

    n_estimators =  np.arange(1, 51, 1)
    params = { 'CLF__n_estimators': n_estimators,
                'CLF__learning_rate': [(2**x)/100 for x in range(7)]+[1]}

    clf = ensemble.AdaBoostClassifier(algorithm='SAMME')

    pipe = Pipeline([('Scale', StandardScaler()),
                     ('CLF', clf)])

    cv = exp.basic_results(pipe, 'BOOST', ds_name, x_train, y_train, x_test, y_test, class_list, params)

    exp.make_complexity_curve(cv.best_estimator_, 'BOOST', ds_name, x, y, 'CLF__n_estimators', 'Estimators', n_estimators, 'linear')

