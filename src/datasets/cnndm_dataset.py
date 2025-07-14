from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from src.datasets.huggingface_dataset import HuggingFaceDataset

class CNNDMHuggingFaceDataset(TorchDataset):
    def __init__(
            self,
            streaming=False,
            split='train',
            shuffle=True,
            shuffle_seed=42,
            max_samples=None,
            model_type="enc-dec",
            filter_max_length=False,
            max_length=1024,
            model_name="t5-base",
            **kwargs
    ):
        super().__init__()
        self.dataset = HuggingFaceDataset(
            path="cnn_dailymail",
            name="3.0.0",
            streaming=streaming,
            split=split,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            max_samples=max_samples
        )

        self.model_type = model_type
        self._preprocess()

        if filter_max_length:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.filter_max_length(max_length)

    def __len__(self):
        return len(self.dataset)

    def _preprocess(self):
        items = []
        for sample in self.dataset:
            prefix = "summarize: "
            inputs = prefix + sample["article"]
            labels = sample["highlights"]

            item = {"text": inputs, "answer": labels}
            if self.model_type == "enc-dec":
                item["input"] = inputs
                item["target"] = labels
            else:
                full_text = inputs + " " + labels
                item["input"] = full_text
                item["target"] = full_text
            items.append(item)
        self.dataset = items

    def filter_max_length(self, max_length):
        dataset_size = len(self.dataset)
        def _keep(sample):
            return len(self.tokenizer(sample["input"], truncation=False)["input_ids"]) <= max_length
        self.dataset = [s for s in self.dataset if _keep(s)]
        print(f'Filtered dataset by total input_ids max_length="{max_length}", size reduced from {dataset_size} to {len(self.dataset)} samples!')

    def __getitem__(self, idx):
        return self.dataset[idx] 