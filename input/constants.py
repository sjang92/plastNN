"""
Frequency Feature Constants
"""
LENGTH = 50  # Length of the TP Sequence to look at
INDEX = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19
}  # Mapping from each RNA to its integer index
NUM_RNA = 20

"""
RNA Interval Feature Constants
"""
RNA_INTERVAL_FEATURE_COLUMNS = ["hr" + str(5 * i) for i in range(1, 9)]
