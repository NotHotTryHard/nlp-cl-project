import evaluate
from src.base import BaseMetric

class AverageRougeMetric(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.rouge = evaluate.load('rouge')
        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, model, batch):
        target, preds = batch['target'], batch['preds']
        rouge_score = self.rouge.compute(references=target, predictions=preds)

        rouge1 = rouge_score["rouge1"].mid.fmeasure
        rouge2 = rouge_score["rouge2"].mid.fmeasure
        rougeL = rouge_score["rougeL"].mid.fmeasure
        return (rouge1 + rouge2 + rougeL) / 3
