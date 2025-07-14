from sklearn.metrics import accuracy_score, f1_score
from src.base.base_metric import BaseMetric

import numpy as np

def map_to_labels(string_labels, task_name):
    if task_name == "stsb":
        return [float(label) for label in string_labels]
    
    label_mappings = {
        "ax": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "cola": {0: "unacceptable", 1: "acceptable"},
        "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "mnli_matched": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "mnli_mismatched": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "mrpc": {0: "not_equivalent", 1: "equivalent"},
        "qnli": {0: "entailment", 1: "not_entailment"},
        "qqp":  {0: "not_duplicate", 1: "duplicate"},
        "rte":  {0: "entailment", 1: "not_entailment"},
        "sst2": {0: "negative", 1: "positive"},
        "wnli": {0: "not_entailment", 1: "entailment"},
    }
    reverse_label_mappings = {
        task: {v: k for k, v in label_mappings[task].items()}
        for task in label_mappings
    }
    return [reverse_label_mappings[task_name].get(label, -1) for label in string_labels]

def f1_score_with_invalid(targets, predictions):
  targets, predictions = np.asarray(targets), np.asarray(predictions)
  # Get indices of invalid predictions
  invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
  # For any prediction != 0 or 1, set it to the opposite of what the target is
  predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
  return 100 * f1_score(targets, predictions)

class CategoricalAccuracy_GLUEMetric(BaseMetric):
    def __init__(self, name=None, model_type="enc-dec", *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.model_type = model_type
        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, model, batch):
        if ('answer' not in batch) or ('task_name' not in batch):
            return 0.0
        answers, predictions = batch['answer'], batch['preds']
        if self.model_type == "dec":
            predictions = [pred.split("answer:")[-1].strip() for pred in predictions]
        predictions = map_to_labels(predictions, batch['task_name'][0])
        answers = map_to_labels(answers, batch['task_name'][0])
        return accuracy_score(answers, predictions)
    
class F1_GLUEMetric(BaseMetric):
    def __init__(self, name=None, model_type="enc-dec", *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.model_type = model_type
        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, model, batch):
        if ('answer' not in batch) or ('task_name' not in batch):
            return 0.0
        answers, predictions, task_name = batch['answer'], batch['preds'], batch['task_name'][0]
        if task_name not in ["qqp", "mrpc"]:
            return 0.0
        if self.model_type == "dec":
            predictions = [pred.split("answer:")[-1].strip() for pred in predictions]
        predictions = map_to_labels(predictions, batch['task_name'][0])
        answers = map_to_labels(answers, batch['task_name'][0])
        return f1_score_with_invalid(answers, predictions)
