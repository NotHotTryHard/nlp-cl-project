from sklearn.metrics import accuracy_score, f1_score
from src.base.base_metric import BaseMetric

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
        answers, predictions, task_name = batch['answer'], batch['preds'], batch['task_name']
        if task_name not in ["qqp", "mrpc"]:
            return 0.0
        if self.model_type == "dec":
            predictions = [pred.split("answer:")[-1].strip() for pred in predictions]
        return f1_score(answers, predictions, average='macro')
