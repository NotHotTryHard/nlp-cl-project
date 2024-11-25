from src.base import BaseMetric
import re

class ArithmeticAccuracyMetric(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, model, batch):
        target, preds = batch['target'], batch['preds']
        res = []
        for i in range(len(preds)):
            pred_number = extract_number_from_pred(preds[i])
            target_number = extract_number_from_pred(target[i])
            res.append(1.0 if pred_number == target_number else 0.0)
        return sum(res) / len(res)
    
    
class ArithmeticMAEMetric(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, model, batch):
        target, preds = batch['target'], batch['preds']
        res = []
        for i in range(len(preds)):
            pred_number = extract_number_from_pred(preds[i])
            target_number = extract_number_from_pred(target[i])
            res.append(abs(pred_number - target_number))
        return sum(res) / len(res)


def extract_number_from_pred(text):
    match = re.search(r'\d+(\.\d+)?', text)
    if match:
        return float(match.group())
    return 0.0