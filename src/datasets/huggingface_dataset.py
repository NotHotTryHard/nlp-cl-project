from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset

class HuggingFaceDataset(TorchDataset):
    def __init__(
            self,
            path,
            name,
            streaming=False,
            split=None,
            data_files=None
            ):
        super().__init__()
        self.dataset = load_dataset(
            path,
            name,
            streaming=streaming,
            split=split,
            data_files=data_files
        )
        self.streaming=streaming

    def __len__(self):
        if self.streaming:
            return float('inf')
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)
