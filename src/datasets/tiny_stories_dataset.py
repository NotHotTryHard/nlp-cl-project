import json
import os
import numpy as np
import torch

from pathlib import Path
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Union, List, Tuple
from tqdm import tqdm

from src.utils import ROOT_PATH


class TinyStoriesDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42

    def __init__(self, 
                 raw_data_dir,
                 data_dir, 
                 tokenizer_config,
                 val_size = 0.1,
                 max_length = 512,
                 max_index_length=None,
                 train = True):
        """
        Class for dealing with TinyStories Dataset, applying sentencepiece tokenizer
        It processes all initial files, splitting them into train and val directories.
        
        Since it supports both train and val indices, you need to create
        collate_fn for evaluation - default __get_item__ works on training data
        """
        self.train = train
        
        self.raw_data_dir = Path(ROOT_PATH / raw_data_dir)
        self.data_dir = Path(ROOT_PATH / data_dir)
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
            raise StopIteration
            self.process_raw_files()
        else:
            print(f"Found processed dataset files in {self.data_dir}")
        
        model_prefix = ROOT_PATH / tokenizer_config["model_prefix_name"]
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
        self.create_index()

        for token_id in ["pad_id", "unk_id", "bos_id", "eos_id"]:
            setattr(self, token_id, getattr(self.sp_model, token_id)())
        
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

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
    
    def create_index(self):
        print(f"Creating index with SentencePiece...")
        index_type = "train" if self.train else "val"
        with open(self.data_dir / index_type, 'r', encoding='utf-8') as f:
            if not self.max_index_length:
                self.index = self.sp_model.encode(f.readlines())
            else:
                print(f"Truncating to {self.max_index_length} samples")
                files = f.readlines()
                if len(files) > self.max_index_length:
                    files = files[:self.max_index_length]
                self.index = self.sp_model.encode(files)

    def __len__(self):
        return len(self.index)

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
        indices = self.index[item]

        if len(indices) > self.max_length - 2:
            indices = indices[:self.max_length - 2]

        pads = [self.pad_id for _ in range(self.max_length - 2 - len(indices))]
        indices = torch.tensor([self.bos_id] + indices + [self.eos_id] + pads)

        return indices, self.max_length - len(pads)
