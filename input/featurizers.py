
from.constants import *

"""
Helper methods
"""
def _load_train_data_as_dict(file_path, column_names, separator=" ", id_column_name="id"):
    """
    A private utils method that extracts column_names from the given file path.
    the file must be in a csv like format with a designated separator character.

    :param file_path: path to the desired features file
    :param column_names: column names to extract
    :param separator: the separtaor charcater used to format the file
    :return:  dictionary mapping from protein name to the feature
    """

    d = {}

    with open(file_path, 'r') as fp:
        # read first lines for column name parsing, create column index
        columns = fp.readline().rstrip("\n").split(separator)
        col_index = {columns[col]: col for col in range(len(columns))}
        id_col_index = col_index[id_column_name]
        target_columns = [int(col_index[col_name]) for col_name in column_names]

        for line in fp.readlines():
            parsed = line.rstrip("\n").split()
            protein_name = parsed[id_col_index]
            features = [float(parsed[target_column]) for target_column in target_columns]
            d[protein_name] = features

        fp.close()

    return d


"""
Feature Extraction methods
"""
def get_amino_acid_freq_features(protein):
    """
    Featurizes a given protein based on its amino acid frequency.
    Given a protein and its rna sequence, returns a vector of size 20 where
    each index holds the frequency of the corresponding rna between
    the first TP and the 50th TP indices.

    :param protein -  a Protein instance. Defined in protein.py
    :return freq - Python list of length 20
    """
    sequence = protein.sequence
    sequence = sequence[protein.tp_start:protein.tp_start + LENGTH]
    freq = [0] * NUM_RNA  # start with 0 frequency

    for amino_acid in sequence:
        freq[INDEX[amino_acid]] += 1

    return freq  # Python list of length 20


def get_column_features_from_file(file_path, column_names):
    """
    Helper method written to be compatible with the dataset class'
    inject_point_feature method.
    It extracts column_names from the given path

    :param file_path: path to the file with features
    :param column_names
    :return _func: function that looks up the given protein's feature vector from d
    """

    d = _load_train_data_as_dict(file_path, column_names)

    def _func(protein):
        return d[protein.name]

    return _func
