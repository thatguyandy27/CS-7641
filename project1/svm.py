from sklearn import svm
import exp
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(x, y, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
             shrinking=True, probability=False, tol=1e-3, max_iter=-1):
    clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                 shrinking=shrinking, probability=probability, tol=tol, max_iter=max_iter)
    clf = clf.fit(x, y)

    return clf


def run_experiment(x, y, x_train, y_train, x_test, y_test, ds_name, class_list):
    kernels = ['rbf', 'poly', 'sigmoid']
    iters = [-1, int((1e6 / x.shape[0]) / .8) + 1]
    C_values = np.arange(0.001, 2.5, 0.25)

    params = {'CLF__kernel': kernels, 'CLF__max_iter': iters, 'CLF__C': C_values}

    clf = svm.SVC()

    pipe = Pipeline([('Scale', StandardScaler()),
                     ('CLF', clf)])

    cv = exp.basic_results(pipe, 'SVC', ds_name, x_train, y_train, x_test, y_test, class_list, params)

    exp.make_complexity_curve(cv.best_estimator_, 'SVC', ds_name, x, y, 'CLF__kernel', 'Kernels', kernels, 'linear')

