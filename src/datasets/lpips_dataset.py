import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataloader
from tqdm import tqdm

import src.datasets


class IndexedDataset(TorchDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx


class IndexedCollateClass:
    def __init__(self, collate):
        self.collate = collate
    
    def __call__(self, dataset_items):
        indices = [item[-1] for item in dataset_items]
        output = self.collate(dataset_items)
        output.update({"indices": indices})
        return output


class LPIPSReorderedDataset(TorchDataset):
    def __init__(self, dataset_type, dataset_args, embedding_specs, **kwargs):
        super().__init__()

        self.dataset = getattr(src.datasets, dataset_type)(dataset_args)

        self.dataset_surprise_scores = None
        self.prev_mean = None
        self.prev_std = None

        self.model_part = embedding_specs.get("model_part", "encoder")
        self.model_block = embedding_specs.get("model_block", -1)
        self.block_layer = embedding_specs.get("block_layer", -1)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        for tensor_for_gpu in ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch
    
    def _collect_activations_mean_std(self, model, dataset, batch_size, collate, max_samples):
        model.eval()

        dataloader = TorchDataloader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)

        activations_mean = torch.zeros(dataloader.batch_size, model.config.d_model)
        activations_squared_mean = torch.zeros(dataloader.batch_size, model.config.d_model)

        def hook_fn(module, input, output):
            activations_mean += output.detach().mean(dim=1)
            activations_squared_mean += (output ** 2).detach().mean(dim=1)

        # Attach the hook to a specific layer
        model_layer = getattr(model, self.model_part).block[self.model_block].layer[self.block_layer]
        model_layer.register_forward_hook(hook_fn)

        N_batches = max_samples // dataloader.batch_size

        for batch in tqdm(dataloader, len=N_batches):
            batch = self.move_batch_to_device(batch, model.device)
            # collect model embeddings
            model(batch)
        
        activations_mean = activations_mean / N_batches
        activations_std = torch.sqrt(activations_squared_mean / N_batches - activations_mean ** 2)
        return activations_mean, activations_std

    def collect_initial_dataset_activations_mean(self, model, initial_dataset, batch_size, collate, max_samples):
        self.prev_mean, self.prev_std = self._collect_activations_mean(model, initial_dataset, batch_size, collate, max_samples)

    def compute_surprise_scores(self, model, batch_size, collate, max_samples):
        model.eval()

        indexed_dataset = IndexedDataset(self.dataset)
        indexed_collate = IndexedCollateClass(collate)
        dataloader = TorchDataloader(indexed_dataset, batch_size=batch_size, collate_fn=indexed_collate, shuffle=True)
        
        surprise_scores = {}

        def hook_fn(module, input, output):
            activations_mean = output.detach().mean(dim=1)
            
            for ind, activation_mean in zip(batch_indices, activations_mean):
                diff = (activation_mean - self.prev_mean).abs().sum()
                surprise_scores[ind] = diff if diff > self.prev_std.sum() else 0.

        # Attach the hook to a specific layer
        model_layer = getattr(model, self.model_part).block[self.model_block].layer[self.block_layer]
        model_layer.register_forward_hook(hook_fn)

        N_batches = max_samples // batch_size
        for batch, batch_indices in tqdm(dataloader, len= N_batches):
            batch = self.move_batch_to_device(batch, model.device)
            model(batch)
        
        return surprise_scores

    def reorder_dataset(self, model, batch_size, collate, max_samples):
        surprise_scores = self.compute_surprise_scores(model, batch_size, collate, max_samples)
        self.dataset_surprise_scores = sorted(list(surprise_scores.items()), key=lambda x: x[1])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_idx = self.dataset_surprise_scores[idx]
        return self.dataset[original_idx]
