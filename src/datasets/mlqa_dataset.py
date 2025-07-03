from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from src.datasets import HuggingFaceDataset


class MLQAHuggingFaceDataset(TorchDataset):
    def __init__( 
            self,
            name="mlqa.en.en",
            huggingface_split="test", # No train available, we split test manually
            streaming=False,
            shuffle=True,
            shuffle_seed=42,
            split="train",
            split_train_val_test=True,
            split_random_state=42,
            val_size=0.1,
            test_size=0.1,
            **kwargs
    ):
        super().__init__()
        self.dataset = HuggingFaceDataset(
            path="facebook/mlqa",
            name=name,
            streaming=streaming,
            split=huggingface_split,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed
        )

        self.split = split
        self.split_train_val_test = split_train_val_test
        self.split_random_state = split_random_state
        self.val_size = val_size
        self.test_size = test_size

    def _train_test_split(self):
        train_plus_val, test = train_test_split(
            self.dataset,
            test_size=self.test_size,
            random_state=self.split_random_state
        )
        rel_val_size = self.val_size / (1.0 - self.test_size)
        train, val = train_test_split(
            train_plus_val,
            test_size=rel_val_size,
            random_state=self.split_random_state
        )
        if self.split == "train":
            self.dataset = train
        elif self.split == "val":
            self.dataset == val
        else:
            self.dataset = test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
