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

        has_reoredered_datasets = False

        # create and join datasets
        base_dataset = None
        base_dataset_config = None
        datasets = []
        for ds in params["datasets"]:
            dataset = configs.init_obj(
                ds, src.datasets,
                config_parser=configs,
                num_workers=num_workers
            )
            if ds.get("base_dataset", False):
                base_dataset = dataset
                base_dataset_config = ds
            else:
                datasets.append(dataset)
            
            if isinstance(dataset, src.datasets.LPIPSReorderedDataset):
                has_reoredered_datasets = True
        
        assert len(datasets)
        if len(datasets) > 1:
            if params.get("use_mixed_dataset", False):
                ds = params["mixed_dataset"]
                ds["args"]["datasets"] = datasets
                ds["args"]["base_dataset"] = base_dataset
                ds["args"]["base_dataset_config"] = base_dataset_config
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
    
        if has_reoredered_datasets:
            shuffle = False

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"
        
        tokenizer_info = params.get("tokenizer_name", ["T5Tokenizer", "google-t5/t5-small"])
        tokenizer = getattr(transformers, tokenizer_info[0]).from_pretrained(tokenizer_info[1])
        collate_obj = CollateClass(
            tokenizer=tokenizer,
            max_length=params["max_length"],
            mlm_items=params.get("t5_mlm_masking", False), # for training it's generally set in MixedSequentialDataset.update_epoch
            mlm_probability=params.get("mlm_probability", 0.15),
            mean_span_length=params.get("mean_span_length", 3.)
        )
    
        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_obj,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler
        )
        dataloaders[split] = dataloader
    return dataloaders
