import math

from .performance_table import PerformanceTableEntry

def _mcc(tp, fp, fn, tn):
    """
    Helper method for computing Mathews Correlation Coefficient
    :param tp: True Positive
    :param fp: False Positive
    :param fn: False Negative
    :param tn: Truer Negative
    :return: mcc
    """
    mcc = 0.0
    try:
        mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Handle divide by zero
    except Exception:
        pass

    finally:
        return mcc


def _precision_and_recall(tp, fp, fn):
    """
    Helper method for computing precision and recall given the necessary parameters`:w

    :param tp: True Positive
    :param fp: False Positive
    :param fn: False Negative
    :return: ppv, recall pair
    """
    recall, ppv = 0.0, 0.0
    try:
        recall = float(tp) / (tp + fn)
        ppv = float(tp) / (tp + fp)

    # Handle divide by zero
    except Exception:
        pass

    finally:
        return ppv, recall

def evaluate(model, dataset, fold):
    """
    Helper method for evaluating the model on the given dataset's 'fold' cross-validation fold
    :param model: Some model that implements the models.base_neural_network interface
    :param dataset: Some dataset object
    :param fold: cross-validation fold
    :return: evaluation metrics for both train and test data
    """
    stats = [0, 0, 0, 0]  # each being tp, fp, fn, tn
    num_total = 0
    num_corr = 0
    stats_train = [0, 0, 0, 0]
    num_total_train = 0
    num_corr_train = 0

    # Evaluate over the test dataset (Unseen)
    while not dataset.is_testset_over(fold):
        data_dict = dataset.get_next_test(fold)
        input = data_dict["input"]
        label = data_dict["label"]
        num_total += 1
        eval_label = model.evaluate(input, label)
        if eval_label == 0 or eval_label == 3:
            num_corr += 1
        stats[eval_label] += 1

    dataset.reset_testset(fold)  # only reset test set. train set don't need to be reset

    # Evaluate over the train dataset (Seen)
    while not dataset.is_trainset_over(fold):
        data_dict = dataset.get_next_train(fold, should_increment=False)
        input = data_dict["input"]
        label = data_dict["label"]
        num_total_train += 1
        eval_label = model.evaluate(input, label)
        if eval_label == 0 or eval_label == 3:
            num_corr_train += 1
        stats_train[eval_label] += 1

    # Basic confusion matrix stats
    tp_test, fp_test, fn_test, tn_test = stats
    tp_train, fp_train, fn_train, tn_train = stats_train

    # Compute Accuracy, MCC, PPV, Recall
    mcc_test = _mcc(tp_test, fp_test, fn_test, tn_test)
    mcc_train = _mcc(tp_train, fp_train, fn_train, tn_train)
    ppv_train, recall_train = _precision_and_recall(tp_train, fp_train, fn_train)
    ppv_test, recall_test = _precision_and_recall(tp_test, fp_test, fn_test)
    acc_test = float(num_corr) / num_total
    acc_train = float(num_corr_train) / num_total_train

    return PerformanceTableEntry(tp_test, fp_test, fn_test, tn_test, acc_test, mcc_test, ppv_test, recall_test,
                                 tp_train, fp_train, fn_train, tn_train, acc_train, mcc_train, ppv_train, recall_train)
