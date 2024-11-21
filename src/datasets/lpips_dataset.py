import json
import os
import numpy as np
import torch

from datasets import Dataset, DatasetDict
from pathlib import Path
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from typing import Union, List, Tuple
from tqdm import tqdm

from src.utils import ROOT_PATH
from src.datasets.cl_datasets import get_dataset


class LPIPSSequentialDataset(TorchDataset):
    def __init__(self,
                 base_mixing_rate=0.0,
                 sequential_mixing_rate=0.0,
                 base_dataset=None,
                 datasets=None,
                 **kwargs):
        super().__init__()

        self.base_dataset = base_dataset
        self.sequential_datasets = datasets

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

    def update_epoch(self, epoch, epochs):
        if epoch - self.dataset_start_epoch + 1 > epochs // self.num_datasets():
            self.current_dataset += 1
            self.replacements_counter = 0
            self.dataset_start_epoch = epoch
            print(f"Switched to dataset {self.current_dataset + 1} / {self.num_datasets()}")
            return True
        return False
    
    def _order_dataset(self, dataset_idx):
        pass

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
        
        return self.sequential_datasets[n_prev_datasets][real_idx] # (real_idx - prev_length) % len(self.current_dataset)]

