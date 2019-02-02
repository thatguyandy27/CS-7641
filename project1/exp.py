import datetime
import numpy as np
import sklearn.model_selection as ms
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import compute_sample_weight
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
from sklearn import tree

seed = 27
np.random.seed(seed)


def balanced_accuracy(truth, pred):
    # wts = compute_sample_weight('balanced', truth)
    # , sample_weight=wts
    return accuracy_score(truth, pred)


def f1_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return f1_score(truth, pred, average="macro", sample_weight=wts)


scorer = make_scorer(balanced_accuracy)
f1_scorer = make_scorer(f1_accuracy)


# CLF is the classifier
# clf is the classifier name for output
# ds is the data source name for output
def basic_results(clf, clf_name, ds_name, training_x, training_y, test_x, test_y, class_list, params, draw_tree=False):

    curr_scorer = scorer

    # output the grid search results.
    cv = ms.GridSearchCV(clf, n_jobs=1, param_grid=params, refit=True, verbose=10, cv=5, scoring=curr_scorer)
    cv.fit(training_x, training_y)
    reg_table = pd.DataFrame(cv.cv_results_)
    reg_table.to_csv('./output/{}_{}_reg.csv'.format(clf_name, ds_name), index=False)
    test_score = cv.score(test_x, test_y)

    # get the best parameters
    best_estimator = cv.best_estimator_.fit(training_x, training_y)
    final_estimator = best_estimator._final_estimator
    best_params = pd.DataFrame([final_estimator.get_params()])
    best_params.to_csv('./output/{}_{}_best_params.csv'.format(clf_name, ds_name), index=False)

    if draw_tree:
        tree.export_graphviz(final_estimator, out_file='./output/images/{}_{}_LC.dot'.format(clf_name, ds_name))

    test_y_predicted = cv.predict(test_x)
    cnf_matrix = confusion_matrix(test_y, test_y_predicted)
    np.set_printoptions(precision=2)
    plt = plot_confusion_matrix(cnf_matrix, class_list,
                                title='Confusion Matrix: {} - {}'.format(clf_name, ds_name))
    plt.savefig('./output/images/{}_{}_confusion_matrix.png'.format(clf_name, ds_name), format='png', dpi=150,
                bbox_inches='tight')

    plt = plot_confusion_matrix(cnf_matrix, class_list, normalize=True,
                                title='Normalized Confusion Matrix: {} - {}'.format(clf_name, ds_name))

    plt.savefig('./output/images/{}_{}_confusion_matrix_normalized.png'.format(clf_name, ds_name), format='png', dpi=150,
                bbox_inches='tight')


    with open('./output/test results.csv', 'a') as f:
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        f.write('"{}",{},{},{},"{}"\n'.format(ts, clf_name, ds_name, test_score, cv.best_params_))

    n = training_y.shape[0]

    train_sizes = np.append(np.linspace(0.05, 0.1, 20, endpoint=False),
                            np.linspace(0.1, 1, 20, endpoint=True))

    train_sizes, train_scores, test_scores = ms.learning_curve(
        cv.best_estimator_,
        training_x,
        training_y,
        cv=5,
        train_sizes=train_sizes,
        verbose=10,
        scoring=curr_scorer,
        n_jobs=1,
        random_state=seed)

    curve_train_scores = pd.DataFrame(index=train_sizes, data=train_scores)
    curve_test_scores = pd.DataFrame(index=train_sizes, data=test_scores)

    curve_train_scores.to_csv('./output/{}_{}_LC_train.csv'.format(clf_name, ds_name))
    curve_test_scores.to_csv('./output/{}_{}_LC_test.csv'.format(clf_name, ds_name))
    plt = plot_learning_curve('Learning Curve: {} - {}'.format(clf_name, ds_name),
                              train_sizes,
                              train_scores, test_scores)
    plt.savefig('./output/images/{}_{}_LC.png'.format(clf_name, ds_name), format='png', dpi=150)


    return cv

def make_complexity_curve(clf, clf_name, ds_name, x, y,
                          param_name, param_display_name,
                          param_values, x_scale, verbose=False,
                          balanced_dataset=False, threads=1):
    curr_scorer = scorer
    # if not balanced_dataset:
    #     curr_scorer = f1_scorer

    train_scores, test_scores = validation_curve(clf, x, y, param_name, param_values, cv=5, verbose=verbose,
                                                 scoring=curr_scorer, n_jobs=threads)

    curve_train_scores = pd.DataFrame(index=param_values, data=train_scores)
    curve_test_scores = pd.DataFrame(index=param_values, data=test_scores)
    curve_train_scores.to_csv('./output/{}_{}_{}_MC_train.csv'.format(clf_name, ds_name, param_name))
    curve_test_scores.to_csv('./output//{}_{}_{}_MC_test.csv'.format(clf_name, ds_name, param_name))
    plt = plot_model_complexity_curve(
        'Model Complexity: {} - {} ({})'.format(clf_name, ds_name, param_display_name),
        param_values,
        train_scores, test_scores, x_scale=x_scale,
        x_label=param_display_name)
    plt.savefig('./output/images/{}_{}_{}_MC.png'.format(clf_name, ds_name, param_name), format='png',
                dpi=150)

# ------------- PLOTTING CODE ----------#

def plot_model_complexity_curve(title, train_sizes, train_scores, test_scores, ylim=None, multiple_runs=True,
                                x_scale='linear', y_scale='linear',
                                x_label='Training examples (count)', y_label='Accuracy (0.0 - 1.0)',
                                x_ticks=None, x_tick_labels=None, chart_type='line'):
    """
    Generate a simple plot of the test and training model complexity curve.

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    train_sizes : list, array
        The training sizes

    train_scores : list, array
        The training scores

    test_scores : list, array
        The testing sizes

    multiple_runs : boolean
        If True, assume the given train and test scores represent multiple runs of a given test (the default)

    x_scale: string
        The x scale to use (defaults to None)

    y_scale: string
        The y scale to use (defaults to None)

    x_label: string
        Label fo the x-axis

    y_label: string
        Label fo the y-axis
    """
    plt.close('all')
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()

    train_points = train_scores
    test_points = test_scores

    ax = plt.gca()
    if x_scale is not None or y_scale is not None:
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    # TODO: https://stackoverflow.com/questions/2715535/how-to-plot-non-numeric-data-in-matplotlib
    if x_ticks is not None and x_tick_labels is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    train_scores_mean = None
    train_scores_std = None
    test_scores_mean = None
    test_scores_std = None
    if multiple_runs:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        train_points = train_scores_mean
        test_points = test_scores_mean

    if chart_type == 'line':
        if multiple_runs:
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="salmon")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2, color="skyblue")

        plt.plot(train_sizes, train_points, 'o-', linewidth=1, markersize=4,
                 label="Training score")
        plt.plot(train_sizes, test_points, 'o-', linewidth=1, markersize=4,
                 label="Cross-validation score")
    if chart_type == 'bar':

        # https://matplotlib.org/examples/api/barchart_demo.html

        ind = train_sizes
        if x_tick_labels is not None:
            ind = np.arange(len(x_tick_labels))
            ax.set_xticklabels(x_tick_labels)

        bar_width = 0.35
        ax.bar(ind, train_points, bar_width, yerr=train_scores_std, label="Training score")
        ax.bar(ind + bar_width, test_points, bar_width, yerr=test_scores_std,
                        label="Cross-validation score")

        ax.grid(which='both')
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.set_xticks(ind + bar_width / 2)
        if x_tick_labels is not None:
            ax.set_xticklabels(x_tick_labels, rotation=45)

    plt.legend(loc="best")
    plt.tight_layout()

    return plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param cm: The matrix from metrics.confusion_matrics
    :param classes: The classes for the dataset
    :param normalize: If true, normalize
    :param title: The title for the plot
    :param cmap: The color map to use

    :return: The confusion matrix plot
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.close()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def plot_learning_curve(title, train_sizes, train_scores, test_scores, ylim=None, multiple_runs=True,
                        x_scale='linear', y_scale='linear',
                        x_label='Training examples (count)', y_label='Accuracy (0.0 - 1.0)'):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    train_sizes : list, array
        The training sizes

    train_scores : list, array
        The training scores

    test_scores : list, array
        The testing sizes

    multiple_runs : boolean
        If True, assume the given train and test scores represent multiple runs of a given test (the default)

    x_scale: string
        The x scale to use (defaults to None)

    y_scale: string
        The y scale to use (defaults to None)

    x_label: string
        Label fo the x-axis

    y_label: string
        Label fo the y-axis
    """
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()

    train_points = train_scores
    test_points = test_scores

    if x_scale is not None or y_scale is not None:
        ax = plt.gca()
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    if multiple_runs:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        train_points = train_scores_mean
        test_points = test_scores_mean

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2)

    plt.plot(train_sizes, train_points, 'o-', linewidth=1, markersize=4,
             label="Training score")
    plt.plot(train_sizes, test_points, 'o-', linewidth=1, markersize=4,
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
