from torch.utils.data import Dataset as TorchDataset
from src.datasets.huggingface_dataset import HuggingFaceDataset


class MathQADataset(TorchDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = HuggingFaceDataset(**kwargs)
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
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = HuggingFaceDataset(**kwargs)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        input, target = data["question"], data["answer"]
        return [input, target]
