import math



class VoteInfo(object):
    """
    Helper class for voting counts and labeling.
      - pos : number of positive votes
      - neg : number of negative votes
      - pos_probs : list where each index contains a single model's postivie prob given to this VoteInfo
      - neg_probs : same for neg
    """

    def __init__(self):
        self.pos = 0
        self.neg = 0
        self.pos_probs = []
        self.neg_probs = []

    def get_verdict(self):
        """
        Returns the verdict based on the vote counts
        """
        sign = self.pos - self.neg
        if sign == 0:
            return 0
        elif sign > 0:
            return 1
        else:
            return -1

    def to_string(self):
        return "{},{},{},{},{}\n".format(self.get_verdict(), self.pos, self.neg, self.pos_probs, self.neg_probs)

    @staticmethod
    def save_as_csv(name_to_votes, file_path):
        """
        Saves the given dictionary of VoteInfo objects to a csv file
        :param name_to_votes: dictionary of protein_name : VoteInfo object
        """
        output_file = open(file_path, 'w')

        # Write column names
        output_file.write("Verdict,positive votes,negative votes,positive probs,negative probs")

        # Write each row
        for name, vote_info in name_to_votes.items():
            entry = "{},{}".format(name, vote_info.to_string())
            output_file.write(entry)
        output_file.close()

def _mcc(stats):
    """
    Helper method for computing Mathews Correlation Coefficient
    :param stats: tp, fp, fn, tn
    :return: mcc
    """
    tp = stats[0]
    fp = stats[1]
    fn = stats[2]
    tn = stats[3]

    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def evaluate(model, dataset, fold):
    stats = [0, 0, 0, 0]  # each being tp, fp, fn, tn
    num_total = 0
    num_corr = 0
    stats_train = [0, 0, 0, 0]
    num_total_train = 0
    num_corr_train = 0

    while not dataset.is_testset_over(fold):
        data_dict = dataset.get_next_test(fold)
        input = data_dict["input"]
        label = data_dict["label"]
        num_total += 1
        eval_label = model.evaluate(input, label)
        if eval_label == 0 or eval_label == 3:
            num_corr += 1
        stats[eval_label] += 1

    while not dataset.is_trainset_over(fold):
        data_dict = dataset.get_next_train(fold, should_increment=False)
        input = data_dict["input"]
        label = data_dict["label"]
        num_total_train += 1
        eval_label = model.evaluate(input, label)
        if eval_label == 0 or eval_label == 3:
            num_corr_train += 1
        stats_train[eval_label] += 1

    try:
        mcc_value = _mcc(stats)
    except Exception as e:
        mcc_value = 0.0

    try:
        mcc_train = _mcc(stats_train)
    except Exception as e:
        mcc_train = 0.0

    # only reset test set. train set don't need to be reset
    dataset.reset_testset(fold)

    tp = stats[0]
    fp = stats[1]
    fn = stats[2]
    tn = stats[3]

    tp_train = stats_train[0]
    fp_train = stats_train[1]
    fn_train = stats_train[2]
    tn_train = stats_train[3]

    try:
        recall = float(tp) / (tp + fn)
        ppv = float(tp) / (tp + fp)
    except Exception as e:
        recall = ppv = 0.0

    try:
        recall_train = float(tp_train) / (tp_train + fn_train)
        ppv_train = float(tp_train) / (tp_train + fp_train)
    except Exception as e:
        recall_train = ppv_train = 0.0

    return (tp, fp, fn, tn, float(num_corr) / num_total, mcc_value, ppv, recall), \
           (tp_train, fp_train, fn_train, tn_train, float(num_corr_train)/num_total_train, mcc_train, ppv_train, recall_train)
