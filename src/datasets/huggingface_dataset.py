from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset

class HuggingFaceDataset(TorchDataset):
    def __init__(
            self,
            path,
            name=None,
            streaming=False,
            split=None,
            data_files=None,
            max_samples=None,
            shuffle=None,
            shuffle_seed=None
            ):
        super().__init__()

        if max_samples is not None:
            assert not streaming
            assert split is not None

            dataset = load_dataset(path, name=name)[split]
            if shuffle:
                if shuffle_seed is not None:
                    dataset = dataset.shuffle(shuffle_seed)
                else:
                    dataset.shuffle()
            
            self.dataset = dataset.select(range(max_samples))
        else:
            self.dataset = load_dataset(
                path,
                name=name,
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

    def __getitem__(self, idx):
        return self.dataset[idx]
