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


class MixedSequentialDataset(TorchDataset):
    TRAIN_VAL_RANDOM_SEED = 42

    def __init__(self,
                 base_mixing_rate,
                 sequential_mixing_rate=0.0,
                 base_dataset=None,
                 datasets=None,
                 base_dataset_name=None,
                 datasets_names=None,
                 tokenizer=None,
                 args=None,
                 **kwargs):
        """
            Dataset class for sequential fine-tuning on mixed data batches.
            Provides samples from initial dataset / from previous sequential datasets
            with specified probabilities.

            base_mixing_rate: rate of samples from base dataset
            sequential_mixing_rate: rate of samples drawn randomly from previous sequential datasets
            -- SHOULD WE SUPPORT ADATPIVE SEQUENTIAL MIXING RATES?
            
            base_dataset: dataset object, if provided
            datasets: list of sequential datasets as objects, if provided
            OR
            base_dataset_name: the name of the dataset to import from the supported list (get_datasets)
            datasets_names: list of sequential datasets' names to import from the supported list (get_datasets)

            tokenizer: tokenizer for get_datasets, used only in case of dataset names
            args: args for get_datasets, used only in case of dataset names
        """
        super().__init__()

        self.base_dataset = get_dataset(base_dataset_name, tokenizer, args) if base_dataset is None else base_dataset
        self.sequential_datasets = [
            get_dataset(dataset_name, tokenizer, args)
            for dataset_name in datasets_names
        ] if datasets is None else datasets

        self.base_mixing_rate = base_mixing_rate
        self.sequential_mixing_rate = sequential_mixing_rate

        self.cum_sequential_length = np.cumsum([self._get_len(dataset) for dataset in self.sequential_datasets])
        self.replacements_counter = 0
    
    def __len__(self):
        return sum(self._get_len(dataset) for dataset in self.sequential_datasets)

    @staticmethod
    def _get_len(dataset):
        if hasattr(dataset, "num_rows"):
            return dataset.num_rows
        return len(dataset)

    def __getitem(self, idx):
        # SHOULD WE GO FOR THIS ONE?
        # real_idx = (1 - self.base_mixing_rate - self.sequential_mixing_rate)

        p_base = np.random.rand()
        if p_base < self.base_mixing_rate:
            self.replacements_counter += 1
            base_idx = np.random.randint(0, self._get_len(self.base_dataset))
            return self.base_dataset[base_idx]
    
        real_idx = idx - self.replacements_counter
        n_prev_datasets = next(i for i, length in enumerate(self.cum_sequential_length) if length >= real_idx)
        prev_length = self.cum_sequential_length[n_prev_datasets - 1] if n_prev_datasets else 0

        if self.sequential_mixing_rate and n_prev_datasets:
            p_seq = np.random.rand()
            if p_seq < self.sequential_mixing_rate:
                self.replacements_counter += 1
                dataset_idx = np.random.randint(0, n_prev_datasets)
                seq_idx = np.random.randint(0, self._get_len(self.sequential_datasets[dataset_idx]))
                return self.sequential_datasets[dataset_idx][seq_idx]
        
        return self.sequential_datasets[n_prev_datasets][real_idx - prev_length]

class TinyStoriesDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42

    def __init__(self, 
                 raw_data_dir,
                 data_dir, 
                 tokenizer_config,
                 val_size = 0.1,
                 max_length = 512,
                 max_index_length=None,
                 train = True,
                 *args,
                 **kwargs):
        """
        Class for dealing with TinyStories Dataset, applying sentencepiece tokenizer
        It processes all initial files, splitting them into train and val directories.
        
        Since it supports both train and val indices, you need to create
        collate_fn for evaluation - default __get_item__ works on training data
        """
        self.train = train
        
        # self.raw_data_dir = Path(ROOT_PATH / raw_data_dir)
        # self.data_dir = Path(ROOT_PATH / data_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(exist_ok=True, parents=True)
        
        self.raw_files_list = list(os.listdir(self.raw_data_dir))
        raw_files_train, raw_files_val = train_test_split(
            self.raw_files_list,
            test_size=val_size,
            random_state=self.TRAIN_VAL_RANDOM_SEED
        )
        self.raw_files_dict = {"train": raw_files_train, "val": raw_files_val}
        if len(os.listdir(self.data_dir)) == 0:
            self.process_raw_files()
        else:
            print(f"Found processed dataset files in {self.data_dir}")
        
        # model_prefix = ROOT_PATH / tokenizer_config["model_prefix_name"]
        model_prefix = tokenizer_config["model_prefix_name"]
        if not os.path.isfile(model_prefix + '.model'):
            print(f"No pretrained SentencePiece tokenizer found, started training...")
            SentencePieceTrainer.train(
                input=str(self.data_dir / "train"),
                model_prefix=model_prefix,
                **tokenizer_config
            )
        # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=model_prefix + '.model')
        self.vocab_size = tokenizer_config["vocab_size"]
        
        self.max_index_length = max_index_length
        
        for token_id in ["pad_id", "unk_id", "bos_id", "eos_id"]:
            setattr(self, token_id, getattr(self.sp_model, token_id)())
        
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()
        
        self.load_files()
        
    def process_raw_files(self):
        for k, raw_files_list in self.raw_files_dict.items():
            with open(self.data_dir / k, 'a+', encoding='utf-8') as out_file:
                tqdm_desc = f"Processing {k} raw TinyStories..."
                for filename in tqdm(raw_files_list, desc=tqdm_desc):
                    with open(self.raw_data_dir / filename, 'r', encoding='utf-8') as raw_file:
                        data = json.load(raw_file)
                        assert type(data) == list, f"json loaded data type: {type(data)}"
                        for sample in data:
                            out_file.write(sample["story"] + "\n")
    
    def create_raw_index(self):
        if len(os.listdir(self.data_dir)) == 0:
            self.process_raw_files()
        self.raw_index = {"train": [], "val": []}
        for k in ["train", "val"]:
            with open(self.data_dir / k, 'r', encoding='utf-8') as f:
                self.raw_index[k] = f.readlines()
    
    def load_files(self):
        print(f"Loading files...")
        files_type = "train" if self.train else "val"
        with open(self.data_dir / files_type, 'r', encoding='utf-8') as f:
            files = f.readlines()
            if self.max_index_length:
                print(f"Truncating to {self.max_index_length} samples")
                files = files[:self.max_index_length]
        print(f"Done!")
        self.files = files

    def __len__(self):
        return len(self.files)

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :param train: use train indices
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        indices = self.sp_model.encode(self.files[item])

        if len(indices) > self.max_length - 2:
            indices = indices[:self.max_length - 2]

        pads = [self.pad_id for _ in range(self.max_length - 2 - len(indices))]
        indices = torch.tensor([self.bos_id] + indices + [self.eos_id] + pads)

        return indices, self.max_length - len(pads)
