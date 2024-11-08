import argparse
import collections
import warnings

import numpy as np
import torch

from tqdm import tqdm


import src.metric as module_metric
import src.model as module_arch
from src.logger import get_visualizer
from src.utils import prepare_device, MetricTracker
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

def move_batch_to_device(batch, device):
    for tensor_for_gpu in ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]:
        batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
    return batch

def main(config):
    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["model"], module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    print(f"\nNumber of model parameters: {get_number_of_parameters(model)}\n")

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

    inference_indices = {}
    for split in config["data"]:
        if split != "train":
            if config["data"][split].get("inference_on_evaluation", False):
                inference_indices[split] = config["data"][split].get("inference_indices", [24, 2, 22])
    
    evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
    model.eval()

    with torch.no_grad():
        for split, dataloader in evaluation_dataloaders.items():
            losses = []
            metrics_lists = {met.name: [] for met in metrics}
            print(f"Working on {split} validation dataset...")

            for batch_idx, batch in tqdm(enumerate(dataloaders), desc=split, total=len(dataloader)):
                batch = move_batch_to_device(batch, device)
                batch["loss"] = model(batch)
                losses.append(batch["loss"])
                for met in metrics:
                    metrics_lists[met.name].append(met(model, batch))
            
            loss = torch.mean(losses)
            metrics_stats = {name: torch.mean(stats) for name, stats in metrics_lists.items()}
            print(f"Loss on {split}: {loss}")
            for name, stat in metrics_stats.items():
                print(f"Metric {name}: {stat}")
            print()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"
        ),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
