# Don't forget to support cases when target_text == ''
import editdistance
from datasets import load_metric

from src.base import BaseMetric

class RougeMetric(BaseMetric):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.rouge = load_metric('rouge')
        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, **batch):
        rouge_score = self.rouge.compute(references=batch["labels"], predictions=batch["preds"])
        return rouge_score["rouge1"].mid.fmeasure, rouge_score["rouge2"].mid.fmeasure, rouge_score["rougeL"].mid.fmeasure


def calc_cer(target_text: str, predicted_text: str) -> float:
    if not target_text: return int(len(predicted_text) > 0)
    return editdistance.eval(target_text, predicted_text) / len(target_text)

def calc_wer(target_text: str, predicted_text: str) -> float:
    if not target_text: return int(len(predicted_text) > 0)
    target_splitted, predict_splitted = target_text.split(' '), predicted_text.split(' ')
    return editdistance.eval(target_splitted, predict_splitted) / len(target_splitted)
