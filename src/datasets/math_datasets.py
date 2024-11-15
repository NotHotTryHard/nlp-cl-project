from torch.utils.data import Dataset as TorchDataset
from src.datasets.huggingface_dataset import HuggingFaceDataset


class MathQADataset(TorchDataset):
    def __init__(
            self,
            path,
            name=None,
            streaming=False,
            split=None,
            data_files=None,
            max_samples=None,
            shuffle=None,
            shuffle_seed=None,
            **kwargs
    ):
        super().__init__()
        self.dataset = HuggingFaceDataset(path, name, streaming, split, data_files, max_samples, shuffle, shuffle_seed)
        self.category_sep = " [CAT] "
        self.options_sep = " [OPT] "
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        input = data["category"] + self.category_sep + data["Problem"] + self.options_sep + data["options"]
        target = data["correct"] + self.options_sep + data["annotated_formula"] + self.formula_sep + data["Rationale"]
        return [input, target]


class MathDataset(TorchDataset):
    def __init__(
            self,
            path,
            name=None,
            streaming=False,
            split=None,
            data_files=None,
            max_samples=None,
            shuffle=None,
            shuffle_seed=None,
            **kwargs
    ):
        super().__init__()
        self.dataset = HuggingFaceDataset(path, name, streaming, split, data_files, max_samples, shuffle, shuffle_seed)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        input, target = data["question"], data["answer"]
        return [input, target]
