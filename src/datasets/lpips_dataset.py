import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataloader
from tqdm import tqdm

import src.datasets
from src.utils.util import check_cuda_memory, clear_cuda_cache


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
        if "decoder_input_ids" in batch:
            batch["decoder_input_ids"] = batch["decoder_input_ids"].to(device)
        return batch
    
    def _collect_activations_mean_std(self, model, dataset, batch_size, collate, max_samples):
        model.eval()

        dataloader = TorchDataloader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)

        class HookClass:
            def __init__(self):
                super().__init__()
                self.activations_mean = None
                self.activations_squared_mean = None
                self.n_samples = 0

            def __call__(self, module, input, output):
                per_sample_mean = output.detach().mean(dim=1)
                per_sample_squared_mean = (output ** 2).detach().mean(dim=1)
                self.n_samples += per_sample_mean.shape[0]

                if self.activations_mean is None:
                    self.activations_mean = per_sample_mean.sum(dim=0)
                    self.activations_squared_mean = per_sample_squared_mean.sum(dim=0)
                else:
                    self.activations_mean += per_sample_mean.sum(dim=0)
                    self.activations_squared_mean += per_sample_squared_mean.sum(dim=0)
                
            def get_final_activations_stats(self):
                self.activations_mean = self.activations_mean / self.n_samples
                self.activations_squared_mean = self.activations_squared_mean / self.n_samples
                return self.activations_mean.clone(), self.activations_squared_mean.clone()

            def kill_yourself(self):
                del self.activations_mean
                del self.activations_squared_mean
            
        hook = HookClass()

        # Attach the hook to a specific layer
        model_layer = getattr(model.model, self.model_part).block[self.model_block].layer[self.block_layer]
        handle = model_layer.register_forward_hook(hook)

        n_batches = max_samples // dataloader.batch_size
        n_samples = 0

        for batch in tqdm(dataloader, total=n_batches):
            batch = self.move_batch_to_device(batch, model.model.device)
            model(batch)

            n_samples += batch["input_ids"].shape[0] # as batch-size may vary in T5 masked language modelling
            if max_samples <= n_samples:
                break
        
        handle.remove()

        activations_mean, activations_squared_mean = hook.get_final_activations_stats()
        activations_std = torch.sqrt(activations_squared_mean - activations_mean ** 2)
        
        print("After initial, hook not deleted yet:")
        check_cuda_memory()

        hook.kill_yourself()
        
        print("After initial, hook deleted:")
        check_cuda_memory()

        clear_cuda_cache()
        print("After initial, CUDA cache cleared:")
        check_cuda_memory()
        
        return activations_mean, activations_std

    def collect_initial_dataset_activations_mean(self, model, initial_dataset, batch_size, collate, max_samples):
        clear_cuda_cache()
        print("Before initial:")
        check_cuda_memory()
        
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
        print("Before surprise")
        check_cuda_memory()
        
        surprise_scores = self.compute_surprise_scores(model, batch_size, collate, max_samples)

        print("After surprise")
        check_cuda_memory()
        
        self.dataset_surprise_scores = [x[0] for x in sorted(list(surprise_scores.items()), key=lambda x: x[1])]

        del self.prev_mean
        del self.prev_std

        clear_cuda_cache()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_idx = self.dataset_surprise_scores[idx]
        return self.dataset[original_idx]
