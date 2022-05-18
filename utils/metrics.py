class Metrics:
    def __init__(self):
        self.description = "Evaluation Metrics"

    def confusion_matrix(self, tp, tn, fp, fn):
        true_positive_rate = tp / (tp+fn)
        true_negative_rate = tn / (tn+fp)
        false_positive_rate = fp / (fp+tn)
        false_negative_rate = fn / (fn+tp)
        return true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate

    def accuracy(self, tp, tn, fp, fn):
        return float((tp+tn))/float((tp+tn+fp+fn))

    def f1_score(self, tp, fp, fn):
        return float(tp)/float(tp+0.5*(fp+fn))