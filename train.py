import argparse
import collections
import warnings

import numpy as np
import torch

import src.metric as module_metric
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)
    print('Dataloaders are done!')
    # build model architecture, then print to console
    model = config.init_obj(config["model"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    all_params = get_number_of_parameters(model)
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"\nNumber of model parameters: {all_params:_}")
    print(f"Number of trainable parameters: {trainable_params:_}")
    print(f"Trainable: {100.0 * trainable_params / all_params:.2f}%\n")

    # get function handles of loss and metrics
    if hasattr(dataloaders["train"].dataset, "pad_id"):
        pad_id = dataloaders["train"].dataset.pad_id
    else:
        pad_id = -100
    criterion = config.init_obj(config["loss"], torch.nn, ignore_index=pad_id).to(device)

    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    inference_indices = {}
    for split in config["data"]:
        if split != "train":
            if config["data"][split].get("inference_on_evaluation", False):
                inference_indices[split] = config["data"][split].get("inference_indices", [24, 2, 22])

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
        inference_on_evaluation=len(inference_indices) > 0,
        inference_indices=inference_indices,
        first_epoch_eval_only=config["trainer"].get("first_epoch_eval_only", True),
        eval_adapter_order=config["trainer"].get("eval_adapter_order", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c", "--config", default=None, type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "--data_config", default=None, type=str,
        help="[optional] datasets config file path (default: None, data is taken from the main config)",
    )
    args.add_argument(
        "-r", "--resume", default=None, type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d", "--device", default=None, type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "--bs", "--batch_size", default=None, type=int,
        help="train batch size (default: None)",
    )
    args.add_argument(
        "-t", "--task_type", default=None, type=str,
        help='task type for --val_batch_size option to work, supported: ["math", "mlqa"]',
    )
    # args.add_argument(
    #     "--vbs", "--val_batch_size", default=None, type=int,
    #     help="batch size for all ~~hardcoded by task type~~ val datasets (default: None)",
    # )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"
        ),
        CustomArgs(
            ["--val_batch_size"], type=int, target=None
        )
    ]

    hardcoded_val_names = {
        "math": [
            "val_add_or_sub", "val_add_or_sub_in_base", "val_add_or_sub_multiple",
            "val_div", "val_mixed", "val_mul", "val_mul_div_multiple",
            "val_nearest_integer_root", "val_simplify_surd"
        ],
        "mlqa": [
            "val.en.en", "val.de.de", "val.es.es", "val.ar.ar",
            "val.zh.zh", "val.vi.vi", "val.hi.hi"
        ]
    }

    config = ConfigParser.from_args(args, options, hardcoded_val_names)
    main(config)
