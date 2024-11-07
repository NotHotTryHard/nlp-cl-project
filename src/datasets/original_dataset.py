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


class OriginalDataset(TorchDataset):
    def __init__(self,
                 dataset_dir=None,
                 max_length=512,
                 max_samples=None,
                 split="train",
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

        self.max_length = max_length
        self.max_samples = max_samples
        self.split = split
        self.data_dir = ROOT_PATH / dataset_dir
        self.data = self._get_data()
    
    def _get_data(self):
        data = []
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)

            if os.path.isdir(subdir_path):
                file_path = os.path.join(subdir_path, f"{self.split}.txt")
                if not os.path.isfile(file_path):
                    print(f"No such file: {file_path}!")
                    continue
                with open(file_path, 'r') as file:
                    for line in file:
                        input, target = line.strip().split('\t')
                        data.append([input, target])
                        if self.max_samples and len(data) >= self.max_samples:
                            return data
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ returns [input, target] in sentences, tokenizer in collate"""
        return self.data[idx]
    
        # for sentence in self.data[idx]:
            # indices_list.append(self.tokenizer.batch_encode_plus(
            #     [sentence],
            #     padding=False,
            #     max_length=self.max_length,
            #     truncation=True,
            #     return_tensors="pt"
            # ))
        # return indices_list[0], indices_list[1]
