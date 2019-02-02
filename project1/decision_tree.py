from sklearn import tree
import exp
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(data_x, data_y, criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                    presort=False):

    clf = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                      random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                      min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                                      class_weight=class_weight, presort=presort)

    clf = clf.fit(data_x, data_y, None, None, None)

    return clf


def run_experiment(x, y, x_train, y_train, x_test, y_test, ds_name, class_list, max_depth=None):

    if not max_depth:
        max_depth = np.arange(1, x_train.shape[1] * 2, 1)

    params = {'CLF__criterion': ['gini', 'entropy'], 'CLF__max_depth': max_depth,
              'CLF__class_weight': ['balanced', None]}  # , 'DT__max_leaf_nodes': max_leaf_nodes}

    clf = tree.DecisionTreeClassifier()

    pipe = Pipeline([('Scale', StandardScaler()),
                     ('CLF', clf)])

    cv = exp.basic_results(pipe, 'DT', ds_name, x_train, y_train, x_test, y_test, class_list, params)

    exp.make_complexity_curve(cv.best_estimator_, 'DT', ds_name, x, y, 'CLF__max_depth', 'Max Depth', max_depth, 'linear')

