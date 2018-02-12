import numpy as np
import os
import random
from .protein import Protein
import threading

class BaseProteinDataset(object):
    """
    Class to represent a single protein dataset.
    Use this class for cross-validation, feature selection, etc.
    You can provide manual methods to compute point_features as well
    """

    def __init__(self, positive_path, negative_path, feature_paths):
        """
        For unlabeled dataset, feed either only one of the positive / negative paths.
        :param positive_path: path to file holding positive proteins
        :param negative_path: path to file holding negative proteins
        :param feature_paths: pre-computed features to inject
        """
        self.positive_proteins = {}
        self.negative_proteins = {}
        if positive_path:
            self.positive_proteins = self._load_protein(positive_path, 1)  # dict
        if negative_path:
            self.negative_proteins = self._load_protein(negative_path, 0)  # dict

        self.proteins = {}
        self.proteins.update(self.positive_proteins)
        self.proteins.update(self.negative_proteins)

        assert len(self.proteins) == (self.num_negative() + self.num_positive()),\
            "Union on positive and negative proteins yield a smaller subset. Must have no intersection"

        for feature_path in feature_paths:
            self._inject_feature(self.proteins, feature_path)

    def __getitem__(self, item):
        """
        this method allows us to use any instance of this class and its child classes as
        an enumerable of datasets.

        dataset = BaseProteinDataset(...)
        for protein in dataset:
            # protein is an instance of the Protein calss
        """
        protein_name = list(self.proteins.keys())[item]
        return self.proteins.get(protein_name)

    def add_tp_start(self, tp_file):
        """
        Helper method for adding TP start index per protein.
        :param tp_file: path to file holding the TP start index data
        """
        with open(tp_file, 'r') as fp:
            for line in fp.readlines():
                tup = line.rstrip('\n').split()
                name = tup[0]
                if name == "id": continue
                tp_start = int(tup[1])
                protein = self.proteins[name]
                protein.tp_start = tp_start

    def _load_protein(self, path, label):
        """
        returns a dict of proteins with the give label
        """

        proteins = {}

        with open(path, 'r') as fp:
            for line in fp.readlines():
                tup = line.rstrip("\n").split()  # split on white space
                name = tup[0]
                if name == "id": continue
                sequence = [amino for amino in tup[1]]
                protein = Protein(name, sequence, label)
                proteins[name] = protein

        return proteins

    def _inject_feature(self, proteins, feature_path):
        """
        injects features encoded in the given feature_path into all proteins in our sequence
        raises errors for easy debugging
        """

        head, tail = os.path.split(feature_path)
        feature_name = tail[0:-4]

        with open(feature_path, 'r') as fp:
            for line in fp.readlines():
                data_point = line.rstrip("\n").split()
                name = data_point[0]
                data = [float(val) / 100 for val in data_point[1:]]  # convert all to float
                if name in proteins:
                    if not proteins[name].length == len(data):
                        raise ValueError("Feature length does not equal the sequence length")

                    proteins[name][feature_name] = data  # modify data
                else:
                    raise IOError("Non existent protein is found. Protein name = {}".format(name))

    def inject_point_feature(self, feature_name, feature_func):
        """
        Method for injecting point features into each protein
        We apply feature_func to each protein. feature_func takes in a single parameter,
        which is an instance of the Protein class

        feature_func can emit multiple values (as a list).
        In this case, each feature will have name feature_name_idx
        """
        for name in self.proteins.keys():
            protein = self.proteins[name]
            feature = feature_func(protein)

            if isinstance(feature, list):
                num_values = len(feature)
                for i in range(num_values):
                    feature_i_name = "{}_{}".format(feature_name, str(i))
                    protein[feature_i_name] = feature[i]
            else:
                protein[feature_name] = feature

    def clip_sequence_to_fixed_length(self, start, end):
        """
        Helper method for clipping the rna sequence of all protines in this dataset
        """
        for name in self.proteins.keys():
            protein = self.proteins[name]
            assert end > start and end-start <= protein.length
            for feature in protein.keys():
                protein[feature] = protein[feature][start:end]

    def num_positive(self):
        return len(self.positive_proteins)

    def num_negative(self):
        return len(self.negative_proteins)

    def num_samples(self):
        return len(self.proteins)

    def _cross_validation_sets_naive(self, k=5):
        """
        method for generating indices for cross_validation
        returns a list of length 5 where each index holds a tuple of (train array, test array)
        """
        indices = list(range(self.num_samples()))
        random.shuffle(indices)
        k_folds = np.array_split(indices, k)

        result = []

        for fold in range(k):
            test_set = k_folds[fold]
            train_set = [k_folds[i] for i in range(k) if i != fold]
            train_set_flatten = [idx for sub_fold in train_set for idx in sub_fold]
            result.append(tuple([train_set_flatten, test_set]))

        return result

    def cross_validation_sets_ratio(self, k=5, positive=1, negative=5):
        """
        method for generating indices for cross_validation, where we generate k fold sets
        but we use positive : negative ratio per cross_validation set.
        Since we have almost 1:2 ratio in our dataset, all remainder datapoints will be pushed to the
        test set.
        """
        return None  # TODO: implement


class FeedDictProteinDataset(BaseProteinDataset):
    """
    Subclass of BaseProteinDataset that implements an interface for
    feed-dict based tensorflow usage.

    Since the dataset is very small, we store everything in memory and
    don't worry about I/O costs
    """

    def __init__(self, positive_path, negative_path, feature_paths, k=5):
        self.k = k
        self.cursors = [[0, 0] for _ in range(k)]
        self.epochs = [0 for _ in range(k)]

        # Synchronization (Supports quick training of more complex models. Not used for plastNN)
        self.cursor_lock = threading.Lock()

        super(FeedDictProteinDataset, self).__init__(positive_path, negative_path, feature_paths)

    def generate_records_for_cross_validation(self):
        """
        Method for pre-generating training/testing cv datasets for feed-dict based tensorflow interface.
        we store everything in memory
        """
        cv_sets = self._cross_validation_sets_naive(self.k)
        fold_num = 0
        self.folds = []

        for fold in cv_sets:
            train = fold[0]
            test = fold[1]

            train_data = [{"input": self[idx].get_all_feature_vectors(), "label": self[idx].label} for idx in train]
            test_data = [{"input": self[idx].get_all_feature_vectors(), "label": self[idx].label} for idx in test]
            self.folds.append((train_data, test_data))
            fold_num += 1

    def _inc_cursor(self, fold, num_items, train=True, should_increment=True):
        """
        increments cursor, keep track of epochs etc
        """
        idx = 0 if train else 1
        curr = self.cursors[fold][idx]
        next = curr + 1
        did_exhaust = False

        # Keep track of if we've exhausted or not
        if next >= num_items and train:
            did_exhaust = True
            if should_increment:
                self.epochs[fold] += 1
            next = 0

        self.cursors[fold][idx] = next

        return did_exhaust

    def _get_cursor(self, fold, train=True):
        """
        returns the current cursor
        """
        idx = 0 if train is True else 1
        return self.cursors[fold][idx]

    def get_next_train(self, fold, should_increment=True):
        """
        Queue-like interface for getting datasets in sequence
        """
        self.cursor_lock.acquire()

        num_train = len(self.folds[fold][0])  # 0 for train
        result = self.folds[fold][0][self._get_cursor(fold, True)]
        self._inc_cursor(fold, num_train, True, should_increment=should_increment)  # True for train dataset

        self.cursor_lock.release()
        return result

    def get_next_test(self, fold=0):
        """
        Queue-like interface for getting datasets in sequence (testset)
        """
        self.cursor_lock.acquire()

        num_test = len(self.folds[fold][1])  # 1 for test
        result = self.folds[fold][1][self._get_cursor(fold, False)]
        self._inc_cursor(fold, num_test, False)  # False for test dataset

        self.cursor_lock.release()
        return result

    def is_testset_over(self, fold=0):
        self.cursor_lock.acquire()

        cursor = self._get_cursor(fold, False)
        num_test = len(self.folds[fold][1])
        if cursor >= num_test:
            self.cursor_lock.release()
            return True
        else:
            self.cursor_lock.release()
            return False

    def is_trainset_over(self, fold):
        self.cursor_lock.acquire()
        cursor = self._get_cursor(fold, True)
        num_train = len(self.folds[fold][0])
        if cursor >= num_train - 1:
            self._inc_cursor(fold, num_train, True,should_increment=False)  # this is hacky and not good.
            self.cursor_lock.release()
            return True
        else:
            self.cursor_lock.release()
            return False

    def shuffle_train(self, fold):
        random.shuffle(self.folds[fold][0])

    def get_epoch(self, fold):
        return self.epochs[fold]

    def reset_testset(self, fold=0):
        self.cursor_lock.acquire()
        self.cursors[fold][1] = 0
        self.cursor_lock.release()


class PointProteinDataset(FeedDictProteinDataset):
    """
    Subclass of BaseProteinDataset where we only care about the point features of each protein.
    Pointfeatures are the featuers of a protein that doesn't featurize each aminoacid
    as a sequence, but compresses all information into a single vector
    """
    def generate_records_for_cross_validation(self):
        """
        Method for pre-generating training/testing cv datasets for feed-dict based tensorflow interface.
        we store everything in memory
        """
        cv_sets = self._cross_validation_sets_naive(self.k)

        self.folds = []
        fold_num = 0

        for fold in cv_sets:
            print("generating records for fold = {}".format(fold_num))
            train = fold[0]
            test = fold[1]

            train_data = [{"input": self[idx].get_point_feature_vector(), "label": self[idx].label} for idx in train]
            test_data = [{"input": self[idx].get_point_feature_vector(), "label": self[idx].label} for idx in test]
            print("fold {} has {} training data and {} test data".format(fold_num, len(train_data), len(test_data)))
            self.folds.append((train_data, test_data))
            fold_num += 1

    def generate_records_for_labeling(self):
        """
        Shape this dataset so that it can be consumed for labeling unlabeled data
        """
        print("Generating records for evaluation...")
        self.folds = []  # simply follow cross validation scheme

        num_proteins = len(self.proteins.keys())
        print("Labeling for total {} proteins".format(num_proteins))

        evaluate_set = []
        for protein in self:
            data = {"input": protein.get_point_feature_vector(), "label": protein.label,
                    "name": protein.name}  # preten there's label
            evaluate_set.append(data)
        self.folds.append(([], evaluate_set))

