import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

def compute_dataset_embeddings():
    pass


class LPIPSReorderedDataset(TorchDataset):
    def __init__(self, dataset, prev_embeddings, device, embedding_specs, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.dataset_embeddings = []
        self.prev_embeddings = prev_embeddings
        self.device = device

        self.model_part = embedding_specs.get("model_part", "encoder")
        self.model_block = embedding_specs.get("model_block", -1)
        self.block_layer = embedding_specs.get("block_layer", -1)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        for tensor_for_gpu in ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    
    def compute_dataset_embeddings(self, model, dataloader, max_samples):
        model.eval()
        for batch_idx, batch in enumerate(tqdm(dataloader, len= max_samples // dataloader.batch_size)):
            batch = self.move_batch_to_device(batch, self.device)
            
            # collect model embeddings
            model(batch)

            # DO WE NEED TO SAVE INTERMEDIATE OUTPUTS INSIDE OF HUGGINGFACE MODEL?
            model_layer = getattr(model, self.model_part).block[self.model_block].layer[self.block_layer]
            