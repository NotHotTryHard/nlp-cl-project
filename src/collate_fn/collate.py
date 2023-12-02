import logging
import torch

from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)

class CollateClass:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    
    def __call__(self, dataset_items):
        indices = [item[0] for item in dataset_items]
        lengths = [item[1] for item in dataset_items]

        indices = pad_sequence(indices, padding_value=self.pad_id, batch_first=True)
        lengths = torch.tensor(lengths, dtype=torch.long)
        
        indices = indices[:, :lengths.max()]
        
        return {"indices": indices, "lengths": lengths}
