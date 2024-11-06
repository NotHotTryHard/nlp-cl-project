from datasets import load_metric
from src.base import BaseMetric

class RougeMetric(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.rouge = load_metric('rouge')
        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, **batch):
        rouge_score = self.rouge.compute(references=batch["labels"], predictions=batch["preds"])
        return rouge_score["rouge1"].mid.fmeasure, rouge_score["rouge2"].mid.fmeasure, rouge_score["rougeL"].mid.fmeasure
