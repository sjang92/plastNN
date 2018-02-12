import pickle
import matplotlib.pyplot as plt
import os

class PerformanceTableEntry(object):
    """
    Helper class that represents each evaluation entry stored in the Performance Table
    """

    def __init__(self, tp, fp, fn, tn, acc, mcc, ppv, recall,
                 tp_tr, fp_tr, fn_tr, tn_tr, acc_tr, mcc_tr, ppv_tr, recall_tr):
        self.tp = tp
        self.fp=fp
        self.fn = fn
        self.tn = tn
        self.acc = acc
        self.mcc = mcc
        self.ppv = ppv
        self.recall = recall
        self.tp_tr = tp_tr
        self.fp_tr = fp_tr
        self.fn_tr = fn_tr
        self.tn_tr = tn_tr
        self.acc_tr = acc_tr
        self.mcc_tr = mcc_tr
        self.ppv_tr = ppv_tr
        self.recall_tr = recall_tr

    @staticmethod
    def get_csv_columns():
        return "True Positives, False Positives, False Negatives, True Negatives, Accuracy, MCC, PPV, Recall\n"

    def to_string(self):
        return "{},{},{},{},{},{},{},{}\n"\
            .format(self.tp, self.fp, self.fn, self.tn, self.acc, self.mcc, self.ppv, self.recall)

    def report(self):
        print("Test : Accuracy = {} / mcc = {} / ppv = {} / recall = {}"
              .format(self.acc, self.mcc, self.ppv, self.recall))
        print("Train : Accuracy = {} / mcc = {} / ppv {} / recall = {}"
              .format(self.acc_tr, self.mcc_tr, self.ppv_tr, self.recall_tr))


class PerformanceTable(object):
    """
    Helper class for recording the evaluation performance of our models
    over multiple folds and epochs
    """

    def __init__(self, run_id, k, epochs):
        self.run_id = run_id
        self.k = k
        self.epochs = epochs
        self.table = {}
        self.path = "./{}/".format(run_id)

        # Insert empty lists per fold
        for i in range(k):
            self.table[i] = [None] * self.epochs

    def insert_raw_data(self, fold, epoch, tp, fp, fn, tn, acc, mcc, ppv, recall,
                        tp_tr, fp_tr, fn_tr, tn_tr, acc_tr, mcc_tr, ppv_tr, recall_tr):
        """
        Use for testing
        """
        new_entry = PerformanceTableEntry(tp, fp, fn, tn, acc, mcc, ppv, recall,
                                          tp_tr, fp_tr, fn_tr, tn_tr, acc_tr, mcc_tr, ppv_tr, recall_tr)
        self.table[fold][epoch] = new_entry

    def insert(self, entry, fold, epoch):
        """
        Insert the given PerformanceTableEntry at the fold/epoch pair
        :param entry: PerformanceTableEntry object holding the evaluation results
        :param fold: cross-validation fold
        :param epoch: epoch number for the given fold
        """
        self.table[fold][epoch] = entry

    def get_best_mcc(self, fold):
        """
        Returns the best mcc value in the given fold
        :param fold: cross-validation fold number
        :return: best mcc in the given cross-validation fold
        """
        entries = self.table[fold]
        best_mcc = 0.0
        for entry in entries:
            if entry != None and entry.mcc >= best_mcc:
                best_mcc = entry.mcc
        return best_mcc

    def get_best_acc(self, fold):
        """
        Returns the best acc value in the given fold
        :param fold: cross-validation fold number
        :return: best acc in the give ncross-validation fold
        """
        entries = self.table[fold]
        best_acc = 0.0
        for entry in entries:
            if entry != None and entry.acc >= best_acc:
                best_acc = entry.acc

        return best_acc

    @classmethod
    def load_from_pickle(cls, path):
        """
        Use for testing
        """
        fp = open(path, 'r')
        result = pickle.load(fp)
        fp.close()
        return result

    def save_as_pickle(self, path):
        """
        Use for testing
        """
        print("Saving performance table as pickle file at {}".format(path))
        fp = open(path, 'w')
        pickle.dump(self, fp)
        fp.close()
        print("Finished saving performance table")

    def save_as_csv(self, path):
        """
        Saves this PerformanceTable in csv format at the given path
        :param path: file path
        """
        print("Saving the performance table as a csv file at {}".format(path))
        fp = open(path, 'w')
        fp.write("FOLD,EPOCH,{}".format(PerformanceTableEntry.get_csv_columns()))

        for k in range(self.k):
            for epoch in range(self.epochs):
                entry = self.table[k][epoch]
                if entry:
                    fold_epoch = "{},{},".format(k, epoch)
                    fp.write(fold_epoch + entry.to_string())
        fp.close()

    def print_perf(self, fold, epoch):
        print("Performance after fold {} epoch {}".format(fold, epoch))
        self.table[fold][epoch].report()

    def plot(self):

        for fold in range(self.k):
            xs = range(1, len(self.table[fold]) + 1)

            acc_train = [self.table[fold][epoch].acc_tr for epoch in range(len(self.table[fold]))]
            acc_test = [self.table[fold][epoch].acc for epoch in range(len(self.table[fold]))]
            mcc_train = [self.table[fold][epoch].mcc_tr for epoch in range(len(self.table[fold]))]
            mcc_test = [self.table[fold][epoch].mcc for epoch in range(len(self.table[fold]))]

            plt.plot(xs, acc_train, label="fold {}, train acc".format(fold))
            plt.plot(xs, acc_test, label="fold {}, test acc".format(fold))
            plt.plot(xs, mcc_train, label="fold {}, train mcc".format(fold))
            plt.plot(xs, mcc_test, label="fold {}, test mcc".format(fold))

        plt.savefig()





