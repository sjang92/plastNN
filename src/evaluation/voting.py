class VoteInfo(object):
    """
    Helper class for voting counts and labeling.
      - pos : number of positive votes
      - neg : number of negative votes
      - pos_probs : list where each index contains a single model's postivie prob given to this VoteInfo
      - neg_probs : same for neg
    """
    COLUMN_NAMES = ["Verdict", "Positive Votes", "Negative Votes", "Positive Probilities", "Negative Probabilities"]

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
        pos_probs_string = "/".join([str(val) for val in self.pos_probs])
        neg_probs_string = "/".join([str(val) for val in self.neg_probs])
        return "{},{},{},{},{}\n".format(self.get_verdict(), self.pos, self.neg, pos_probs_string, neg_probs_string)

    @classmethod
    def save_as_csv(cls, name_to_votes, file_path):
        """
        Saves the given dictionary of VoteInfo objects to a csv file
        :param name_to_votes: dictionary of protein_name : VoteInfo object
        """
        print("Saving the voting results as a csv file at {}".format(file_path))

        output_file = open(file_path, 'w')

        # Write column names
        output_file.write("{}\n".format(",".join(cls.COLUMN_NAMES)))

        # Write each row
        for name, vote_info in name_to_votes.items():
            entry = "{},{}".format(name, vote_info.to_string())
            output_file.write(entry)
        output_file.close()
