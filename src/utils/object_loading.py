import transformers

from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import src.datasets
from src import batch_sampler as batch_sampler_module
from src.collate_fn.collate import CollateClass
from src.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            tokenizer_info = ds.get("tokenizer_name", ["T5Tokenizer", "google-t5/t5-small"])
            ds["tokenizer"] = getattr(transformers, tokenizer_info[0]).from_pretrained(tokenizer_info[1])

            datasets.append(configs.init_obj(
                ds, src.datasets,
                config_parser=configs,
                num_workers=num_workers
            ))
        
        assert len(datasets)
        if len(datasets) > 1:
            if params.get("use_mixed_dataset", False):
                ds = params["mixed_dataset"]
                ds["datasets"] = datasets
                dataset = configs.init_obj(
                    ds, src.datasets,
                    config_parser=configs,
                    num_workers=num_workers
                )
            else:
                dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                                             data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"
        
        collate_obj = CollateClass(pad_id=dataset.pad_id)
    
        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_obj,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler
        )
        dataloaders[split] = dataloader
    return dataloaders
