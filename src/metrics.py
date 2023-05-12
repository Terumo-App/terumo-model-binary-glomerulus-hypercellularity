from sklearn import metrics

class Metrics:
    def __init__(self) -> None:
          self.y_true = None
          self.y_pred = None

          self.accuracy = None
          self.precision = None
          self.recall = None
          self.fscore = None

          self.kappa = None
          self.precision_curve = None
          self.recall_curve = None


    def compute_metrics(self, y_true, y_pred):
            self.y_true = y_true
            self.y_pred = y_pred
            self.accuracy = metrics.accuracy_score(y_true, y_pred)
            self.precision, self.recall, self.fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
            self.kappa = metrics.cohen_kappa_score(y_true, y_pred)
    