import copy
import logging
import pandas as pd
import numpy as np

from collections import Counter

from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from abc import ABC, abstractmethod

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_pairplot(title, df, class_column_name=None):
    plt = sns.pairplot(df, hue=class_column_name)
    return plt


# Adapted from https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count / n) * np.log((count / n)) for clas, count in classes])
    return H / np.log(k) > 0.75


class DataLoader(ABC):
    def __init__(self, path, verbose, seed):
        self._path = path
        self._verbose = verbose
        self._seed = seed

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def load_and_process(self, data=None, preprocess=True):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and should be called before any processing is done.

        :return: Nothing
        """
        if data is not None:
            self._data = data
            self.features = None
            self.classes = None
            self.testing_x = None
            self.testing_y = None
            self.training_x = None
            self.training_y = None
        else:
            self._load_data()
        self.log("Processing {} Path: {}, Dimensions: {}", self.data_name(), self._path, self._data.shape)
        if self._verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            self.log("Data Sample:\n{}", self._data)
            pd.options.display.max_rows = old_max_rows

        if preprocess:
            self.log("Will pre-process data")
            self._preprocess_data()

        self.get_features()
        self.get_classes()

        self.log("Feature dimensions: {}", self.features.shape)
        self.log("Classes dimensions: {}", self.classes.shape)
        self.log("Class values: {}", np.unique(self.classes))
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]
        self.log("Class distribution: {}", class_dist)
        self.log("Class distribution (%): {}", (class_dist / self.classes.shape[0]) * 100)
        self.log("Sparse? {}", isspmatrix(self.features))

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)

        self.log("Binary? {}", self.binary)
        self.log("Balanced? {}", self.balanced)

    def scale_standard(self):
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

            # pipe = Pipeline([
            #     ('Scale', StandardScaler()),
            #     ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
            #     ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
            #     ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
            #     ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median'))
            # ])

            # transformed = pipe.fit_transform(self.training_x, self.training_y)
            # print(transformed)
            # print(transformed.shape)

    def create_histograms(self):
        for feature in self._data.columns:
            plt.close()
            plt.figure()
            plt.hist(self._data[feature], color='blue', edgecolor='black')
            # Add labels
            # plt.title()
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('./output/images/{}_{}_histogram.png'.format(self.data_name(), feature),
                        format='png', dpi=150)
        plt.close()
        plt.figure()
        features = [feature for feature in self._data.columns]
        for i in np.arange(18).astype(np.int):
            plt.subplot(3, 6, i + 1)
            plt.hist(self._data[features[i]], color='blue', edgecolor='black')
            plt.xlabel("{}".format(features[i]))
            plt.subplots_adjust(bottom=0.01, right=1.0, top=1.0, left=0, wspace=0, hspace=0)
            # plt.axis('off')
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)  # labels along the bottom edge are off

        plt.savefig('./report/{}_histogram.png'.format(self.data_name()),
                    format='png', dpi=150)

        return

    def build_train_test_split(self, test_size=0.3):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.classes, test_size=test_size, random_state=self._seed, stratify=self.classes
            )

    def get_features(self, force=False):
        if self.features is None or force:
            self.log("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.log("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    def dump_test_train_val(self, test_size=0.2, random_state=123):
        ds_train_x, ds_test_x, ds_train_y, ds_test_y = ms.train_test_split(self.features, self.classes,
                                                                           test_size=test_size,
                                                                           random_state=random_state,
                                                                           stratify=self.classes)
        pipe = Pipeline([('Scale', preprocessing.StandardScaler())])
        train_x = pipe.fit_transform(ds_train_x, ds_train_y)
        train_y = np.atleast_2d(ds_train_y).T
        test_x = pipe.transform(ds_test_x)
        test_y = np.atleast_2d(ds_test_y).T

        train_x, validate_x, train_y, validate_y = ms.train_test_split(train_x, train_y,
                                                                       test_size=test_size, random_state=random_state,
                                                                       stratify=train_y)
        test_y = pd.DataFrame(np.where(test_y == 0, -1, 1))
        train_y = pd.DataFrame(np.where(train_y == 0, -1, 1))
        validate_y = pd.DataFrame(np.where(validate_y == 0, -1, 1))

        tst = pd.concat([pd.DataFrame(test_x), test_y], axis=1)
        trg = pd.concat([pd.DataFrame(train_x), train_y], axis=1)
        val = pd.concat([pd.DataFrame(validate_x), validate_y], axis=1)

        tst.to_csv('./data/{}_test.csv'.format(self.data_name()), index=False, header=False)
        trg.to_csv('./data/{}_train.csv'.format(self.data_name()), index=False, header=False)
        val.to_csv('./data/{}_validate.csv'.format(self.data_name()), index=False, header=False)

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def data_name(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def class_column_name(self):
        pass

    @abstractmethod
    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes

    def pipe(self, clf_label, clf):
        return Pipeline([('Scale', StandardScaler()), (clf_label, clf)])

    def reload_from_hdf(self, hdf_path, hdf_ds_name, preprocess=True):
        self.log("Reloading from HDF {}".format(hdf_path))
        loader = copy.deepcopy(self)

        df = pd.read_hdf(hdf_path, hdf_ds_name)
        loader.load_and_process(data=df, preprocess=preprocess)
        loader.build_train_test_split()

        return loader

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))


class WineData(DataLoader):
    def __init__(self, path='./data/wine/wine.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        # wine_red = pd.read_csv('../data/wine/winequality-red.csv', header=0, sep=";")
        # wine_red['is_red'] = 1
        wine_white = pd.read_csv('./data/wine/winequality-white.csv', sep=";")
        # wine_white['is_red'] = 0

        self._data = wine_white

    def class_column_name(self):
        return 'quality'

    def data_name(self):
        return 'Wine'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class DiabetesRenop(DataLoader):
    def __init__(self, path='./data/diabetes_renop/data.txt', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        # madX1 = pd.read_csv('./data/madelon/madelon_train.data', header=None, sep=' ')
        # madX2 = pd.read_csv('./data/madelon/madelon_valid.data', header=None, sep=' ')
        # madX = pd.concat([madX1, madX2], 0).astype(float)
        # madY1 = pd.read_csv('./data/madelon/madelon_train.labels', header=None, sep=' ')
        # madY2 = pd.read_csv('./data/madelon/madelon_valid.labels', header=None, sep=' ')
        # madY = pd.concat([madY1, madY2], 0)
        # madY.columns = ['Class']
        # mad = pd.concat([madX, madY], 1)
        dataset = pd.read_csv("./data/diabetes_renop/data.txt", header=None)

        # y = dataset.values[:, -1]
        # x = dataset.values[:, :-1]
        self._data = dataset

    def data_name(self):
        return 'Diabetes'

    def class_column_name(self):
        return '19'

    def _preprocess_data(self):
        # self._data = self._data.dropna(axis=1, how='all')
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

if __name__ == '__main__':
    # cd_data = CreditDefaultData(verbose=True)
    # cd_data.load_and_process()

    cw_data = WineData(verbose=True)
    cw_data.load_and_process()

    # co_data = OnlineShoppersData(verbose=True)
    # co_data.load_and_process()
    #
    # cp_data = PenDigitData(verbose=True)
    # cp_data.load_and_process()
