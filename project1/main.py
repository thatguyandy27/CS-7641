import neural_networks
import boosting
import k_nn
import svm
import decision_tree
import numpy as np
import sklearn.model_selection as ms
import pandas as pd
import os



OUTPUT_DIRECTORY = './output'
seed = 27

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))


def run(run_dt, run_knn, run_boost, run_nn, run_svm):
    x_ww, y_ww = load_wine()
    wine_train_x, wine_test_x, wine_train_y, wine_test_y = ms.train_test_split(x_ww, y_ww, test_size=0.25, random_state=seed, stratify=y_ww)
    wine_classes =  np.unique(y_ww)

    x_d, y_d = load_diabetes()
    diabetes_train_x, diabetes_test_x, diabetes_train_y, diabetes_test_y = ms.train_test_split(x_d, y_d, test_size=0.25,
                                                                               random_state=seed, stratify=y_d)
    diabetes_classes = np.unique(y_d)
    
    # x_s, y_s = load_shuttle()
    # shuttle_train_x, shuttle_test_x, shuttle_train_y, shuttle_test_y = ms.train_test_split(x_s, y_s, test_size=0.25,
    #                                                                            random_state=seed, stratify=y_s)
    # shuttle_classes = np.unique(y_s)

    if run_dt:
        decision_tree.run_experiment(x= x_ww, y= y_ww, x_train=wine_train_x, y_train=wine_train_y,
                                     x_test=wine_test_x, y_test=wine_test_y, ds_name='Wine', class_list =wine_classes )

        decision_tree.run_experiment(x=x_d, y=y_d, x_train=diabetes_train_x, y_train=diabetes_train_y,
                                     x_test=diabetes_test_x, y_test=diabetes_test_y, ds_name='Diabetes', class_list=diabetes_classes)

        # decision_tree.run_experiment(x=x_s, y=y_s, x_train=shuttle_train_x, y_train=shuttle_train_y,
        #                              x_test=shuttle_test_x, y_test=shuttle_test_y, ds_name='Shuttle',
        #                              class_list=shuttle_classes)

    if run_knn:
        k_nn.run_experiment(x= x_ww, y= y_ww, x_train=wine_train_x, y_train=wine_train_y,
                                     x_test=wine_test_x, y_test=wine_test_y, ds_name='Wine', class_list =wine_classes )

        k_nn.run_experiment(x=x_d, y=y_d, x_train=diabetes_train_x, y_train=diabetes_train_y,
                                     x_test=diabetes_test_x, y_test=diabetes_test_y, ds_name='Diabetes',
                                     class_list=diabetes_classes)

        # k_nn.run_experiment(x=x_s, y=y_s, x_train=shuttle_train_x, y_train=shuttle_train_y,
        #                              x_test=shuttle_test_x, y_test=shuttle_test_y, ds_name='Shuttle',
        #                              class_list=shuttle_classes)

    if run_boost:
        boosting.run_experiment(x= x_ww, y= y_ww, x_train=wine_train_x, y_train=wine_train_y,
                                     x_test=wine_test_x, y_test=wine_test_y, ds_name='Wine', class_list =wine_classes )

        boosting.run_experiment(x=x_d, y=y_d, x_train=diabetes_train_x, y_train=diabetes_train_y,
                                     x_test=diabetes_test_x, y_test=diabetes_test_y, ds_name='Diabetes',
                                     class_list=diabetes_classes)

        # boosting.run_experiment(x=x_s, y=y_s, x_train=shuttle_train_x, y_train=shuttle_train_y,
        #                     x_test=shuttle_test_x, y_test=shuttle_test_y, ds_name='Shuttle',
        #                     class_list=shuttle_classes)

    if run_nn:
        neural_networks.run_experiment(x= x_ww, y= y_ww, x_train=wine_train_x, y_train=wine_train_y,
                                     x_test=wine_test_x, y_test=wine_test_y, ds_name='Wine', class_list =wine_classes )

        neural_networks.run_experiment(x=x_d, y=y_d, x_train=diabetes_train_x, y_train=diabetes_train_y,
                                     x_test=diabetes_test_x, y_test=diabetes_test_y, ds_name='Diabetes',
                                     class_list=diabetes_classes)

        # neural_networks.run_experiment(x=x_s, y=y_s, x_train=shuttle_train_x, y_train=shuttle_train_y,
        #                         x_test=shuttle_test_x, y_test=shuttle_test_y, ds_name='Shuttle',
        #                         class_list=shuttle_classes)

    if run_svm:
        svm.run_experiment(x= x_ww, y= y_ww, x_train=wine_train_x, y_train=wine_train_y,
                                     x_test=wine_test_x, y_test=wine_test_y, ds_name='Wine', class_list =wine_classes )

        svm.run_experiment(x=x_d, y=y_d, x_train=diabetes_train_x, y_train=diabetes_train_y,
                                     x_test=diabetes_test_x, y_test=diabetes_test_y, ds_name='Diabetes',
                                     class_list=diabetes_classes)

        # svm.run_experiment(x=x_s, y=y_s, x_train=shuttle_train_x, y_train=shuttle_train_y,
        #                         x_test=shuttle_test_x, y_test=shuttle_test_y, ds_name='Shuttle',
        #                         class_list=shuttle_classes)



def load_wine():
    dataset_white_wine = pd.read_csv("./data/wine/winequality-white.csv", sep=';')
    y_ww = dataset_white_wine.values[:, -1]
    x_ww = dataset_white_wine.values[:, :-1]
    return x_ww, y_ww

def load_diabetes():
    dataset = pd.read_csv("./data/diabetes_renop/data.txt", header=None)

    y = dataset.values[:, -1]
    x = dataset.values[:, :-1]
    return x, y

def load_shuttle():
    dataset_shuttle = pd.read_csv("./data/shuttle/shuttle.trn.txt", header=None, sep=' ')
    y = dataset_shuttle.values[:, -1]
    x = dataset_shuttle.values[:, :-1]
    return x, y


def split_data(x, y, test_size=.2):
    return ms.train_test_split(x, y, test_size=test_size, shuffle=True, stratify=y)


if __name__ == '__main__':
    dt_run = True
    knn_run = True
    boost_run = True
    nn_run = True
    # dt_run = False
    # knn_run = False
    # boost_run = False
    # nn_run = False
    svm_run = True
    run(dt_run, knn_run, boost_run, nn_run, svm_run)

