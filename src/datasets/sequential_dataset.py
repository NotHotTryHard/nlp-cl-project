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


class SequentialDataset(TorchDataset):
    def __init__(self,
                 max_length=512,
                 datasets=None,
                 datasets_names=None,
                 tokenizer=None,
                 args=None,
                 **kwargs):
        """
            Dataset class for sequential fine-tuning.

            max_length: max length of the sample
            
            datasets: list of sequential datasets as objects, if provided
            OR
            datasets_names: list of sequential datasets' names to import from the supported list (get_datasets)

            tokenizer: tokenizer for get_datasets, used only in case of dataset names
            args: args for get_datasets, used only in case of dataset names
        """
        super().__init__()

        self.sequential_datasets = [
            get_dataset(dataset_name, tokenizer, args)
            for dataset_name in datasets_names
        ] if datasets is None else datasets

        self.cum_sequential_length = np.cumsum([self._get_len(dataset) for dataset in self.sequential_datasets])
        self.replacements_counter = 0

        self.max_length = max_length
    
    def __len__(self):
        return sum(self._get_len(dataset) for dataset in self.sequential_datasets)

    @staticmethod
    def _get_len(dataset):
        if hasattr(dataset, "num_rows"):
            return dataset.num_rows        НАПИСАТЬ САППОРТ ['train'] и ['text'] сплитов
        return len(dataset)

    def __getitem__(self, idx):
        real_idx = idx - self.replacements_counter
        n_prev_datasets = next(i for i, length in enumerate(self.cum_sequential_length) if length >= real_idx)
        prev_length = self.cum_sequential_length[n_prev_datasets - 1] if n_prev_datasets else 0
        
        sample = self.sequential_datasets[n_prev_datasets][real_idx - prev_length]
    
        if len(sample) > self.max_length - 2:
            sample = sample[:self.max_length - 2]

        pads = [self.pad_id for _ in range(self.max_length - 2 - len(sample))]
        sample = torch.tensor([self.bos_id] + sample + [self.eos_id] + pads)

        return sample
