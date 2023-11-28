import logging
import torch

from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)

class CollateClass:
    def __init__(self, pad_id, max_length):
        self.pad_id = pad_id
        self.max_length = max_length
    
    def __call__(self, dataset_items):
        indices = []
        lengths = []
        
        for item in dataset_items:
            ind, length = item
            indices.append(ind)
            lengths.append(length) 

        indices = pad_sequence(indices, batch_first=True)
        lengths = torch.tensor(lengths, dtype=torch.long)
        
        indices = indices[:, :lengths.max()]
        return {"indices": indices, "lengths": lengths}
