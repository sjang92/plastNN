from __future__ import print_function
import tensorflow as tf
import input.dataset as Dataset
import input.featurizers as Featurizers
import input.constants as DatasetConstants
from models.fully_connected import FullyConnected
from evaluation.performance_table import PerformanceTable
from evaluation.evaluate import evaluate, VoteInfo


"""
Define tensorflow flags
"""
flags = tf.app.flags
flags.DEFINE_string("train_pos_path", "./dataset/train_data/positive.txt", "path to positive proteins")
flags.DEFINE_string("train_neg_path", "./dataset/train_data/negative.txt", "path to negative proteins")
flags.DEFINE_string("train_pos_tp_path", "./dataset/train_data/pos_tp.txt", "path to tp start data for positive proteins")
flags.DEFINE_string("train_neg_tp_path", "./dataset/train_data/neg_tp.txt", "path to tp start data for negative proteins")
flags.DEFINE_string("train_rna_interval_path", "./dataset/train_data/rna.txt", "path to rna interval file")
flags.DEFINE_string("unlabeled_data_path", "./dataset/unlabeled_data/data.txt", "path to unlabeled list of proteins")
flags.DEFINE_string("unlabeled_tp_path", "./dataset/unlabeled_data/tp.txt", "path to tp start data for unlabeled proteins")
flags.DEFINE_string("unlabeled_rna_interval_path", "./dataset/unlabeled_data/rna.txt", "path to rna interval for unlabeled proteins")
flags.DEFINE_integer("k", 6, "number of folds for cross validation")
flags.DEFINE_integer("epochs", 100, "number of passes through the data")
flags.DEFINE_integer("print_every", 500, "number of steps to print")
flags.DEFINE_boolean("should_print_loss", False, "if we should print loss")
flags.DEFINE_string("run_id", "run_id", "id of the run")
flags.DEFINE_boolean("evaluate_every_epoch", True, "If we should evaluate after every epoch")

ARGS = flags.FLAGS

"""
Define Trainer specific helper methods
"""

def prepare_fc_model_dataset(positive_path, negative_path, tp_paths, rna_interval_path, k=1, unlabaled=False):
    """
    Helper function for loading a PointProteinDataset with the given set of parameters
    :param positive_path: path to positive labeled datapoints
    :param negative_path: path to negative labeled datapoints. Can be None if positive_path is for unlabaled data
    :param tp_paths: a list containing paths for tp start data of the concenred proteins
    :param rna_interval_path: a list for rna_interval data
    :param k: cross-validation variable K
    :param unlabaled:  If True, then prepares the dataset object suitable for labeling
    :return: PointProteinDataset object
    """

    # initialize a Protein dataset that featurizes each protein based on its point-wise features
    dataset = Dataset.PointProteinDataset(positive_path, negative_path, [], k)

    # feed tp start data to the dataset
    for path in tp_paths:
        dataset.add_tp_start(path)

    dataset.inject_point_feature("freq", Featurizers.get_amino_acid_freq_features)
    dataset.inject_point_feature("rna_interval",
                                 Featurizers.get_column_features_from_file(rna_interval_path,
                                                                           DatasetConstants.RNA_INTERVAL_FEATURE_COLUMNS))

    # Shape the dataset appropriately for its purpose
    if unlabaled:
        dataset.generate_records_for_evaluation()
    else:
        dataset.generate_records_for_cross_validation()

    return dataset


def evaluate_and_record_results(model, dataset, fold, perf_table, best_models, epoch):
    """
    Helper Method for running evaluation over the dataset with the given model.
    Records the evaluation result in perf_table and saves if required

    :param model: Model object implementing models.base_neurlan_network interface
    :param dataset: a Dataset object
    :param fold: current cross-validation fold
    :param perf_table: PerformanceTable object
    :param best_models: array containing save_path of best models for every cross validation folds
    :param epoch: epoch
    """

    (tp, fp, fn, tn, acc, mcc, ppv, recall), (tp_tr, fp_tr, fn_tr, tn_tr, acc_tr, mcc_tr, ppv_tr, recall_tr) = evaluate(model, dataset, fold)
    perf_table.insert_entry(fold, epoch, tp, fp, fn, tn, acc, mcc, ppv, recall, tp_tr, fp_tr, fn_tr, tn_tr, acc_tr, mcc_tr, ppv_tr, recall_tr)
    perf_table.print_perf(fold, epoch)

    if mcc >= perf_table.get_best_mcc(fold):
        model_path = "{}/fold_{}/tf_save".format(ARGS.run_id, fold)
        best_models[fold] = model_path
        model.save(model_path)


def train_model(model, dataset, fold, perf_table, best_models, evaluate_every_epoch=True):
    """
    Represents a single cross-validation train run for this (model, dataset) pair.
    If evaluate_every_epoch is True, then we run evaluation on the entire train / test
    crossvalidation dataset.

    :param model: Model object implementing models.base_neurlan_network interface
    :param dataset: a Dataset object
    :param fold: current cross-validation fold
    :param perf_table: PerformanceTable object
    :param best_models: array containing save_path of best models for every cross validation folds
    :param evaluate_every_epoch: True or False
    """
    step = 0
    avg_loss = 0.0
    prev_epoch = 0

    while dataset.get_epoch(fold) < ARGS.epochs:
        # extract a single datapoint
        data = dataset.get_next_train(fold)

        # pass the data onto the model and run backpropagation
        logit, one_hot, loss = model.train(data["input"], data["label"])

        avg_loss += loss[0]
        step += 1

        # print loss for monitoring
        if step % ARGS.print_every == 0 and ARGS.should_print_loss:
            print("avg_loss = {}".format(avg_loss / ARGS.print_every))
            avg_loss = 0.0

        # evaluate after every epoch
        curr_epoch = dataset.get_epoch(fold)
        if curr_epoch != prev_epoch and evaluate_every_epoch:
            evaluate_and_record_results(model, dataset, fold, perf_table, best_models, prev_epoch)

        prev_epoch = curr_epoch


def label_with_model(model, unlabeled_dataset, votes):
    """
    Label the unlabeled proteins with the given model.
    The given model updates each VoteInfo object with its vote and score

    :param model: model to assign label with
    :param unlabeled_dataset: PointDataset object holding the unlabeled proteins
    :param votes: dictionary from protein name to VoteInfo objects
    :return: updated votes dictionary
    """

    while not unlabeled_dataset.is_testset_over():

        # get a single protein and assign label with the model
        data_dict = unlabeled_dataset.get_next_test()
        input = data_dict["input"]
        name = data_dict["name"]
        eval_label = model.assign_label(input)

        # create new VoteInfo object if never seen before
        if not (name in votes):
            new_vote_info = VoteInfo()
            votes[name] = new_vote_info
        vote_info = votes[name]

        # add this model's vote and probability to the VoteInfo object
        if eval_label[0][0] == 1:
            vote_info.pos += 1
        else:
            vote_info.neg += 1
        vote_info.pos_probs.append(eval_label[1][0][1])
        vote_info.neg_probs.append(eval_label[1][0][0])

    unlabeled_dataset.reset_testset()

    return votes


if __name__ == "__main__":
    """
    Main Trainer script for PlastNN  
    """

    # 1) Initialize Tensorflow & Helper Objects
    graph = tf.get_default_graph()
    session = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
    perf_table = PerformanceTable(ARGS.run_id, ARGS.k, ARGS.epochs)
    best_models = [None] * ARGS.k

    # 2) Initialize Model and Dataset (Training & Unlabeled)
    model = FullyConnected(session, graph, None, 28, 0.0008)
    training_dataset = prepare_fc_model_dataset(ARGS.train_pos_path, ARGS.train_neg_path,
                                       [ARGS.train_pos_tp_path, ARGS.train_neg_tp_path],
                                       ARGS.train_rna_interval_path, k=ARGS.k)
    unlabeled_dataset = prepare_fc_model_dataset(ARGS.unlabeled_data_path, None,
                                            [ARGS.unlabeled_tp_path],
                                            ARGS.unlabeled_rna_interval_path, unlabaled=True)

    # 3) Perform Training using Cross Validation, and save the results if desired
    for fold in range(ARGS.k):
        model.init()
        train_model(model, training_dataset, fold, perf_table, best_models, evaluate_every_epoch=ARGS.evaluate_every_epoch)

    if ARGS.evaluate_every_epoch:
        perf_table.save_as_csv("{}/perf.csv".format(ARGS.run_id))

    # 4) Label unlabeled_data by K-ways voting and save the results
    votes = {}
    for model_path in best_models:
        model.load(model_path)
        votes = label_with_model(model, unlabeled_dataset, votes)

    vote_file_path = "{}/votes.csv".format(ARGS.run_id)
    VoteInfo.save_as_csv(votes, vote_file_path)

