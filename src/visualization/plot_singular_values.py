import argparse
import math
import matplotlib.pyplot as plt
import transformers
import torch
import torch.nn as nn

import src.model as module_arch
from src.utils import read_json

def plot_singular_values(model, layer_names, sequential=False):
    singular_values_dict = {}
    for i, (name, module) in enumerate(model.named_modules()):
        if name in layer_names:
            svd_lora = model.svd_loras[i]
            if sequential:
                s = torch.cat([svd_lora.s, *[s for s in svd_lora.loras_s]])
            else:
                s = torch.cat([svd_lora.s, svd_lora.lora_s])
            s = s.detach().clone().numpy()
            singular_values_dict[name] = i, s

    num_layers = len(singular_values_dict)
    cols = 3
    rows = math.ceil(num_layers / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for layer_name, (idx, singular_values) in singular_values_dict.items():
        ax = axes[idx]
        original_count = len(model.svd_loras[idx].s)

        ax.plot(singular_values, label="Singular Values")
        ax.axvline(x=original_count - 1, color='red', linestyle='--', label='Separation')

        ax.set_title(layer_name)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()

    for idx in range(len(singular_values_dict), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def create_model(model_config, tokenizer_name="T5Tokenizer", tokenizer_from_pretrained="google-t5/t5-base", max_length=256):
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(tokenizer_from_pretrained)

    module_name = model_config["type"]
    module_args = dict(model_config["args"])
    model = getattr(module_arch, module_name)(**module_args)
    
    if torch.cuda.is_available():
        model = model.to('cuda:0')

    return model, tokenizer

def main():
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    config = read_json(args.config)

    model, tokenizer = create_model(
        model_config=config["model"],
        tokenizer_name=config["tokenizer_name"],
        tokenizer_from_pretrained=config["tokenizer_from_pretrained"]
    )

    plot_singular_values(model, config['layer_names'], sequential="Sequential" in config['model']['type'])


if __name__ == "__main__":
    main()