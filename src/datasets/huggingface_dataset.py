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
            shuffle_seed=None,
            **kwargs
            ):
        super().__init__()

        # Filter kwargs to only include arguments that load_dataset expects
        valid_load_dataset_kwargs = {}
        load_dataset_valid_args = [
            'revision', 'token', 'use_auth_token', 'trust_remote_code', 
            'storage_options', 'streaming', 'num_proc', 'download_config',
            'download_mode', 'verification_mode', 'keep_in_memory', 'save_infos'
        ]
        
        for key, value in kwargs.items():
            if key in load_dataset_valid_args:
                valid_load_dataset_kwargs[key] = value

        if max_samples is not None:
            assert not streaming
            assert split is not None

            dataset = load_dataset(path, name=name, split=split, data_files=data_files, **valid_load_dataset_kwargs)
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
                data_files=data_files,
                **valid_load_dataset_kwargs
            )
        
        self.streaming=streaming

    def __len__(self):
        if self.streaming:
            return int(float('inf'))
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
