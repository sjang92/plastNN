import numpy as np
import collections
from .featurizers import INDEX

class Protein(object):
    """
    Class to represent a single protein as a dict of features
    """

    def __init__(self, name, sequence, label):
        self.name = name
        self.label = np.array([label])  # array for tensorflow compatibility
        self.length = len(sequence)
        self.sequence = sequence
        self.point_features = collections.defaultdict(float)
        self.sequence_features = collections.defaultdict(list)  # per feature, sequence is converted into a list of values
        self.tp_start = None

    """
    Base Protein Methods for Dict-like behavior emulation
    """
    def __setitem__(self, key, item):
        """
        Supports syntax
         protein['feature_name'] = some_value
        """
        if isinstance(item, list):
            # assert len(item) == self.length, "length of a feature must be the same as the sequence length"
            self.sequence_features[key] = item
        else:
            self.point_features[key] = item

    def __getitem__(self, key):
        """
        Supports syntax
         feature = protein['feature_name']
        """
        if key in self.point_features:
            return self.point_features[key]
        else:
            return self.sequence_features[key]

    def __len__(self):
        return len(self.point_features) + len(self.sequence_features)

    def __eq__(self, protein):
        # Use this method only for protein name comparison.
        assert isinstance(protein, Protein) or isinstance(protein, str), \
            "can't compare with an instance of a non-Protein or String[name comparison] class"
        if isinstance(protein, str):
            return self.name == protein
        return self.name == protein.name

    def __ne__(self, protein):
        return not self.__eq__(protein)

    def __hash__(self):
        return hash(self.name)

    """
    Custom getters for Protein specific properties / features
    """
    def keys(self):
        return self.point_features.keys() + self.sequence_features.keys()

    def has_key(self, k):
        return (k in self.point_features) or (k in self.sequence_features)

    def values(self):
        return self.point_features.values() + self.sequence_features.values()

    def items(self):
        return self.point_features.items() + self.sequence_features.items()

    def get_feature_vector_at(self, idx):
        """
        Returns a feature vector of dimension d where d = # of sequence features
        return type is numpy nd array
        """
        if idx >= self.length:
            raise IndexError("Index out of bounds for the given protein. idx = {}, length = {}".format(idx, self.length))
            return
        feature_vec = [feature_value[idx] for feature_name, feature_value in self.sequence_feature_items()]
        return np.array(feature_vec)

    def get_all_feature_vectors(self):
        """
        Returns all feature vectors of all amino acids as a list.
        """
        result = []
        for idx in range(self.length):
            result.append(self.get_feature_vector_at(idx))

        return result

    def get_feature_vectors_between(self, start, end):  # inc, exc
        """
        Returns all features vectors between the given indices, suitable for rnn consumption.
        Doesn't care about index out of bounds. Assumes the caller knows all
        """
        result = []

        for idx in range(start, end):
            result.append(self.get_feature_vector_at(idx))

        return np.array([result])


    def get_point_feature_vector(self):
        """
        returns all point features as a vector
        """
        result = [[value for feature, value in self.point_feature_items()]]
        return np.array(result)

    def point_feature_names(self):
        return self.point_features.keys()

    def point_features(self):
        return self.point_features

    def point_feature_items(self):
        return self.point_features.items()

    def sequence_feature_names(self):
        return self.sequence_features.keys()

    def sequence_features(self):
        return self.sequence_features

    def sequence_feature_items(self):
        return self.sequence_features.items()

    def get_feature_type(self, feature):
        if feature in self.point_features():
            return 'point'
        elif feature in self.sequence_features():
            return 'sequence'
        else:
            return None

    def get_sequence_between(self, start, end):
        result = []
        for idx in range(start, end):
            amino_acid = self.sequence[idx]
            amino_acid_idx = INDEX[amino_acid]
            result.append(amino_acid_idx)

        return np.array([result])
