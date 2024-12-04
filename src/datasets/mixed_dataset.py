import numpy as np

from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

import src.collate_fn
import src.datasets
from src.utils import ROOT_PATH
from src.collate_fn.collate import CollateClass
from src.datasets.lpips_dataset import LPIPSReorderedDataset

class MixedSequentialDataset(TorchDataset):
    def __init__(self,
                 base_mixing_rate=0.0,
                 sequential_mixing_rate=0.0,
                 base_dataset=None,
                 datasets=None,
                 base_dataset_config=None,
                 lpips_base_coeff=1.0,
                 lpips_prev_coeff=1.0,
                 **kwargs):
        """
            Dataset class for sequential fine-tuning on mixed data batches.
            Provides samples from initial dataset / from previous sequential datasets
            with specified probabilities.
            
            Switches between datasets epoch-wise via update_epoch method.

            base_mixing_rate: rate of samples from base dataset
            sequential_mixing_rate: rate of samples drawn randomly from previous sequential datasets
            -- SHOULD WE SUPPORT ADATPIVE SEQUENTIAL MIXING RATES?
            
            base_dataset: dataset object, if provided
            datasets: list of sequential datasets as objects, if provided
        """
        super().__init__()

        self.base_dataset = base_dataset
        self.sequential_datasets = datasets

        self.base_dataset_config = base_dataset_config
        self.initial_collate = None

        self.lpips_base_coeff = lpips_base_coeff if self.base_dataset is not None else 0.0
        self.lpips_prev_coeff = lpips_prev_coeff

        self.base_mixing_rate = base_mixing_rate
        self.sequential_mixing_rate = sequential_mixing_rate

        self.cum_sequential_length = np.cumsum([len(dataset) for dataset in self.sequential_datasets])
        self.replacements_counter = 0

        self.current_dataset = 0
        self.dataset_start_epoch = 1
    
    def __len__(self):
        # return sum(len(dataset) for dataset in self.sequential_datasets)
        return len(self.sequential_datasets[self.current_dataset])
    
    def num_datasets(self):
        return len(self.sequential_datasets)

    def create_base_collate(self, collate):
        return CollateClass(
            tokenizer=collate.tokenizer,
            max_length=collate.max_length,
            mlm_items=True,
            mlm_probability=self.base_dataset_config.get("mlm_probability", 0.15),
            mean_span_length=self.base_dataset_config.get("mean_span_length", 3.)
        )

    def create_regular_collate(self, collate):
        return CollateClass(
            tokenizer=collate.tokenizer,
            max_length=collate.max_length,
            mlm_items=False
        )
    
    def reorder_if_lpips(self, model, batch_size, collate, max_samples, update_scores=False):
        if isinstance(self.sequential_datasets[self.current_dataset], src.datasets.LPIPSReorderedDataset):
            # print(f"Reordering the dataset {self.current_dataset + 1} w.r.t. the mean embeddings diff...")
            prev_dataset = self.sequential_datasets[self.current_dataset - 1]
            dataset = self.sequential_datasets[self.current_dataset]

            dataset.collect_initial_dataset_activations_mean(model, prev_dataset, batch_size, collate, max_samples)
            dataset.reorder_dataset(
                model, batch_size, collate, max_samples,
                update_scores=update_scores,
                alpha=self.lpips_base_coeff,
                beta=self.lpips_prev_coeff
            )
            # print(f"Finished reordering the dataset {self.current_dataset + 1}.")
        
    def reorder_if_lpips_base(self, model, batch_size, initial_collate, collate, max_samples):
        if isinstance(self.sequential_datasets[self.current_dataset], src.datasets.LPIPSReorderedDataset):
            # print(f"Reordering the dataset {self.current_dataset + 1} w.r.t. the mean embeddings diff...")
            dataset = self.sequential_datasets[self.current_dataset]

            # BASE_DATASET SUPPORT
            dataset.collect_initial_dataset_activations_mean(model, self.base_dataset, batch_size, initial_collate, max_samples)
            dataset.reorder_dataset(model, batch_size, collate, max_samples)
            # print(f"Finished reordering the dataset {self.current_dataset + 1}.")

    def update_epoch(self, epoch, epochs, model=None, dataloader=None, max_samples=None):       
        batch_size = dataloader.batch_size
        collate = dataloader.collate_fn

        if epoch - self.dataset_start_epoch + 1 > epochs // self.num_datasets():
            self.current_dataset += 1
            self.replacements_counter = 0
            self.dataset_start_epoch = epoch

            if not isinstance(self.sequential_datasets[self.current_dataset], LPIPSReorderedDataset):
                return True

            print(f"Switched to dataset {self.current_dataset + 1} / {self.num_datasets()}")
            if self.base_dataset is not None:
                print(f"Reordering dataset {self.current_dataset + 1} w.r.t. the base dataset with coeff={self.lpips_base_coeff}...")
                self.reorder_if_lpips_base(model, batch_size, self.initial_collate, collate, max_samples)

                print(f"Updating order of {self.current_dataset + 1} dataset w.r.t. to the previous one with coeff={self.lpips_prev_coeff}...")
                self.reorder_if_lpips(model, batch_size, collate, max_samples, update_scores=True)
            else:
                print(f"Reordering dataset {self.current_dataset + 1} w.r.t. the previous one...")
                self.reorder_if_lpips(model, batch_size, collate, max_samples)
            
            print(f"Finished reordering the dataset {self.current_dataset + 1}.")
            return True
        
        if not isinstance(self.sequential_datasets[self.current_dataset], LPIPSReorderedDataset):
            return False
        
        if epoch == 1 and self.base_dataset is not None:
            self.initial_collate = self.create_base_collate(collate)
            # dataloader.collate_fn = self.initial_collate
            print(f"Reordering dataset {self.current_dataset + 1} w.r.t. the base dataset...")
            self.reorder_if_lpips_base(model, batch_size, self.initial_collate, collate, max_samples)
            print(f"Finished reordering the dataset {self.current_dataset + 1}.")
            return True
        
        return False

    def __getitem__(self, idx):
        if self.base_dataset is not None and self.base_mixing_rate:
            p_base = np.random.rand()
            if p_base < self.base_mixing_rate:
                self.replacements_counter += 1
                base_idx = np.random.randint(0, len(self.base_dataset))
                return self.base_dataset[base_idx]
    
        real_idx = idx - self.replacements_counter
        n_prev_datasets = self.current_dataset
        # n_prev_datasets = next(i for i, length in enumerate(self.cum_sequential_length) if length > real_idx)
        prev_length = self.cum_sequential_length[n_prev_datasets - 1] if n_prev_datasets else 0

        if self.sequential_mixing_rate and n_prev_datasets:
            p_seq = np.random.rand()
            if p_seq < self.sequential_mixing_rate:
                self.replacements_counter += 1
                dataset_idx = np.random.randint(0, n_prev_datasets)
                seq_idx = np.random.randint(0, len(self.sequential_datasets[dataset_idx]))
                return self.sequential_datasets[dataset_idx][seq_idx]
        
        return self.sequential_datasets[self.current_dataset][real_idx] # (real_idx - prev_length) % len(self.current_dataset)]
