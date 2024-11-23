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
        indices = [item[1] for item in dataset_items]
        output = self.collate([item[0] for item in dataset_items])
        output.update({"indices": indices})
        return output


class LPIPSReorderedDataset(TorchDataset):
    def __init__(self, dataset_type, dataset_args, embedding_specs, **kwargs):
        super().__init__()

        self.dataset = getattr(src.datasets, dataset_type)(**dataset_args)

        self.dataset_surprise_scores = list(range(len(self.dataset)))
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

        class HookClass:
            def __init__(self, batch_size, d_model, device):
                super().__init__()
                self.activations_mean = torch.zeros(batch_size, d_model, device=device)
                self.activations_squared_mean = torch.zeros(batch_size, d_model, device=device)

            def __call__(self, module, input, output):
                self.activations_mean += output.detach().mean(dim=1)
                self.activations_squared_mean += (output ** 2).detach().mean(dim=1)
            
        hook = HookClass(batch_size, model.model.config.d_model, model.model.device)

        # Attach the hook to a specific layer
        model_layer = getattr(model.model, self.model_part).block[self.model_block].layer[self.block_layer]
        handle = model_layer.register_forward_hook(hook)

        N_batches = max_samples // dataloader.batch_size
        for batch in tqdm(dataloader, total=N_batches):
            batch = self.move_batch_to_device(batch, model.model.device)
            # collect model embeddings
            model(batch)
        
        handle.remove()
        
        activations_mean = hook.activations_mean / N_batches
        activations_std = torch.sqrt(hook.activations_squared_mean / N_batches - activations_mean ** 2)
        return activations_mean, activations_std

    def collect_initial_dataset_activations_mean(self, model, initial_dataset, batch_size, collate, max_samples):
        self.prev_mean, self.prev_std = self._collect_activations_mean_std(
            model, initial_dataset, batch_size, collate, max_samples
        )

    def compute_surprise_scores(self, model, batch_size, collate, max_samples):
        model.eval()

        indexed_dataset = IndexedDataset(self.dataset)
        indexed_collate = IndexedCollateClass(collate)
        dataloader = TorchDataloader(indexed_dataset, batch_size=batch_size, collate_fn=indexed_collate, shuffle=True)
        
        class HookClass:
            def __init__(self, prev_mean, prev_std):
                super().__init__()
                self.surprise_scores = {}
                self.batch_indices = None
                self.prev_mean = prev_mean
                self.prev_std = prev_std

            def __call__(self, module, input, output):
                activations_mean = output.detach().mean(dim=1)
                for ind, activation_mean in zip(self.batch_indices, activations_mean):
                    diff = (activation_mean - self.prev_mean).abs().sum()
                    self.surprise_scores[ind] = diff if diff > self.prev_std.sum() else 0.
            
        hook = HookClass(self.prev_mean, self.prev_std)        

        # Attach the hook to a specific layer
        model_layer = getattr(model.model, self.model_part).block[self.model_block].layer[self.block_layer]
        handle = model_layer.register_forward_hook(hook)

        N_batches = max_samples // batch_size
        for batch in tqdm(dataloader, total=N_batches):
            batch = self.move_batch_to_device(batch, model.model.device)
            hook.batch_indices = batch['indices']
            model(batch)
        
        surprise_scores = hook.surprise_scores
        handle.remove()
        return surprise_scores

    def reorder_dataset(self, model, batch_size, collate, max_samples):
        surprise_scores = self.compute_surprise_scores(model, batch_size, collate, max_samples)
        self.dataset_surprise_scores = [x[0] for x in sorted(list(surprise_scores.items()), key=lambda x: x[1])]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_idx = self.dataset_surprise_scores[idx]
        return self.dataset[original_idx]
