from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from src.base.base_metric import BaseMetric

class CategoricalAccuracy_GLUEMetric(BaseMetric):
    def __init__(self, name=None, model_type="enc-dec", *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.model_type = model_type
        self.requires_preds = True
        self.compute_on_train = False
        self.metric = CategoricalAccuracy().get_metric(reset=False)
    
    def __call__(self, model, batch):
        # if ('answer' not in batch) or ('lang' not in batch):
            # return 0.0
        answers, predictions = batch['answer'], batch['preds']
        if self.model_type == "dec":
            predictions = [pred.split("answer:")[-1].strip() for pred in predictions]
        return self.metric(predictions, answers)
    

class F1_GLUEMetric(BaseMetric):
    def __init__(self, name=None, model_type="enc-dec", *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.model_type = model_type
        self.requires_preds = True
        self.compute_on_train = False
        prc, rcl, f1 = F1Measure().get_metric(reset=False)
        self.metric = f1
    
    def __call__(self, model, batch):
        # if ('answer' not in batch) or ('lang' not in batch):
            # return 0.0
        answers, predictions = batch['answer'], batch['preds']
        if self.model_type == "dec":
            predictions = [pred.split("answer:")[-1].strip() for pred in predictions]
        return self.metric(predictions, answers)
