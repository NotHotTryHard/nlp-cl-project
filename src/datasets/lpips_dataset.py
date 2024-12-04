import bisect
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
    def __init__(self, dataset_type, dataset_args, embedding_specs, cdf_surprise_scores=True, **kwargs):
        super().__init__()

        self.dataset = getattr(src.datasets, dataset_type)(**dataset_args)

        self.dataset_surprise_scores = [(i, 0) for i in range(len(self.dataset))]
        self.prev_mean = None
        self.prev_std = None
        self.prev_activations = None

        self.model_part = embedding_specs.get("model_part", "encoder")
        self.model_block = embedding_specs.get("model_block", -1)
        self.block_layer = embedding_specs.get("block_layer", -1)

        self.cdf_surprise_scores = cdf_surprise_scores

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

                self.activations = None

            def __call__(self, module, input, output):
                per_sample_mean = output.detach().mean(dim=1)
                per_sample_squared_mean = (output ** 2).detach().mean(dim=1)
                self.n_samples += per_sample_mean.shape[0]

                if self.activations_mean is None:
                    self.activations = per_sample_mean
                    self.activations_mean = per_sample_mean.sum(dim=0)
                    self.activations_squared_mean = per_sample_squared_mean.sum(dim=0)
                else:
                    self.activations = torch.cat((self.activations, per_sample_mean), 0)
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

        with torch.no_grad():
            for batch in tqdm(dataloader, total=n_batches):
                batch = self.move_batch_to_device(batch, model.model.device)
                model(batch)

                n_samples += batch["input_ids"].shape[0] # as batch-size may vary in T5 masked language modelling
                if max_samples <= n_samples:
                    break
        
        handle.remove()

        activations_mean, activations_squared_mean = hook.get_final_activations_stats()
        activations_std = torch.sqrt(activations_squared_mean - activations_mean ** 2)

        activations = hook.activations
        
        print("After initial, hook not deleted yet:")
        check_cuda_memory()

        hook.kill_yourself()
        
        print("After initial, hook deleted:")
        check_cuda_memory()

        clear_cuda_cache()
        print("After initial, CUDA cache cleared:")
        check_cuda_memory()
        
        return activations_mean, activations_std, activations

    def collect_initial_dataset_activations_mean(self, model, initial_dataset, batch_size, collate, max_samples):
        clear_cuda_cache()
        print("Before initial:")
        check_cuda_memory()
        
        self.prev_mean, self.prev_std, self.prev_activations = self._collect_activations_mean_std(
            model, initial_dataset, batch_size, collate, max_samples
        )
    
    def compute_diffs(self, activations):
        min_diff = 1e10
        max_diff = 0.0
        diffs = []

        for activation in activations:
            diffs.append([])
            for prev_activation in self.prev_activations:
                diff = torch.norm(activation - prev_activation)
                diffs[-1].append(diff)
                min_diff = min(min_diff, diff)
                max_diff = max(max_diff, diff)

        diffs = [
            list(sorted([(diff, j) for j, diff in enumerate(diffs[i])]))
            for i in range(len(diffs))
        ]   

        return diffs, min_diff, max_diff

    def get_diffs_numpy(diffs):
        return [[x[0].cpu().numpy() for x in diffs[i]] for i in range(len(diffs))]

    def compute_diffs_cumsums(self, diffs, num_points=100, start=None, end=None):
        if start is None or end is None:
            start, end = 1e10, 0.0
            for diff in diffs:
                start = min(start, min([x[0].item() for x in diff]))
                end = max(end, max([x[0].item() for x in diff]))
            print(start, end)

        log_start = torch.log10(torch.tensor(start))
        log_end = torch.log10(torch.tensor(end))
        diff_grid = torch.logspace(log_start, log_end, steps=num_points).cpu().numpy()

        diffs_numpy = self.get_diffs_numpy(diffs)

        diffs_cumsums = []
        
        for i, diff_numpy in enumerate(diffs_numpy):
            diffs_cumsums.append([])
            for diff_threshold in diff_grid:
                pos = bisect.bisect_left(diff_numpy, diff_threshold)
                diffs_cumsums[-1].append(pos)
            if i % 10 == 0:
                print('passed', i)

        return diffs_cumsums

    def get_diffs_ema(diffs_cumsums):
        diffs_ema = []
        ema_coeffs = torch.logspace(0.0, torch.log10(torch.tensor(0.1)), steps=len(diffs_cumsums[0]))
        for diff_cumsum in diffs_cumsums:
            diffs_ema.append(sum(diff * coeff for diff, coeff in zip(diff_cumsum, ema_coeffs)))
        return diffs_ema

    def compute_surprise_scores(self, model, batch_size, collate, max_samples):
        model.eval()

        indexed_dataset = IndexedDataset(self.dataset)
        indexed_collate = IndexedCollateClass(collate)
        dataloader = TorchDataloader(indexed_dataset, batch_size=batch_size, collate_fn=indexed_collate, shuffle=True)
        
        class HookClass:
            def __init__(self, prev_mean, prev_std, cdf_surprise_scores=False):
                super().__init__()
                self.surprise_scores = {}
                self.batch_indices = None
                self.prev_mean = prev_mean
                self.prev_std = prev_std
                self.activations = None
                self.indices = None
                self.cdf_surprise_scores = cdf_surprise_scores

            def __call__(self, module, input, output):
                activations_mean = output.detach().mean(dim=1)

                if self.cdf_surprise_scores:
                    diffs, min_diff, max_diff = self.compute_diffs(activation_mean)
                    diffs_cumsums = self.compute_diffs_cumsums(diffs, num_points=100, start=0.66 * min_diff, end=1.5 * max_diff)

                    diffs_ema = self.get_diffs_ema(diffs_cumsums)
                    scores = {i: x for i, x in enumerate(diffs_ema)}

                    for score, dataset_ind in zip(scores, self.batch_indices):
                        self.surprise_scores[dataset_ind] = score
                else:
                    for ind, activation_mean in zip(self.batch_indices, activations_mean):
                        diff = (activation_mean - self.prev_mean).abs()
                        self.surprise_scores[ind] = torch.norm(diff)
            
        hook = HookClass(self.prev_mean, self.prev_std, cdf_surprise_scores=self.cdf_surprise_scores)  

        # Attach the hook to a specific layer
        model_layer = getattr(model.model, self.model_part).block[self.model_block].layer[self.block_layer]
        handle = model_layer.register_forward_hook(hook)

        N_batches = max_samples // batch_size
        with torch.no_grad():
            for batch in tqdm(dataloader, total=N_batches):
                batch = self.move_batch_to_device(batch, model.model.device)
                hook.batch_indices = batch['indices']
                model(batch)
        
        surprise_scores = hook.surprise_scores

        handle.remove()
        return surprise_scores

    def reorder_dataset(self, model, batch_size, collate, max_samples, update_scores=False, alpha=None, beta=None):
        print("Before surprise")
        check_cuda_memory()
        
        surprise_scores = self.compute_surprise_scores(model, batch_size, collate, max_samples)

        print("After surprise")
        check_cuda_memory()

        if update_scores is None:
            self.dataset_surprise_scores = list(surprise_scores.items())
        else:
            self.dataset_surprise_scores = [
                (ind, alpha * prev_score + beta * surprise_scores[ind])
                for ind, prev_score in self.dataset_surprise_scores
            ]
        
        self.dataset_surprise_scores = list(sorted(self.dataset_surprise_scores), lambda x: x[1])
        
        del self.prev_mean
        del self.prev_std

        clear_cuda_cache()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_idx = self.dataset_surprise_scores[idx][0]
        return self.dataset[original_idx]
