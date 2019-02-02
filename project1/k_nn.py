from sklearn import neighbors
import exp
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(data_x, data_y, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
          p=2, metric='minkowski', metric_params = None, n_jobs = None):

    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p,
                                 leaf_size=leaf_size, metric=metric, metric_params = metric_params, n_jobs = n_jobs)

    clf = clf.fit(data_x, data_y)

    return clf

def run_experiment(x, y, x_train, y_train, x_test, y_test, ds_name, class_list, max_depth=None):

    neighbor_count =  np.arange(1, 51, 1)
    params = {'CLF__metric': ['manhattan', 'euclidean', 'chebyshev'], 'CLF__n_neighbors': neighbor_count,
              'CLF__weights': ['uniform']}

    clf = neighbors.KNeighborsClassifier()

    pipe = Pipeline([('Scale', StandardScaler()),
                     ('CLF', clf)])

    cv = exp.basic_results(pipe, 'KNN', ds_name, x_train, y_train, x_test, y_test, class_list, params)

    exp.make_complexity_curve(cv.best_estimator_, 'KNN', ds_name, x, y, 'CLF__n_neighbors', 'Neighbor count', neighbor_count, 'linear')

