from src.collate_fn.collate import CollateClass
from src.datasets.lpips_dataset import LPIPSReorderedDataset
from src.model import T5forSummarization
from src.utils import read_json

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import argparse
import numpy as np
import matplotlib.pyplot as plt
import transformers


def collect_activations_by_datasets(model, tokenizer, datasets_configs, embedding_specs, collates, batch_size=8, max_samples=1000):
    activations = {}
    for i, (dataset_config, collate) in enumerate(zip(datasets_configs, collates)):
        lpips_dataset = LPIPSReorderedDataset(**dataset_config, embedding_specs=embedding_specs)
        activations[i] = {}
        activations[i]["mean"], activations[i]["std"], activations[i]["activations"] = \
            lpips_dataset._collect_activations_mean_std(
                model=model,
                dataset=lpips_dataset.dataset,
                batch_size=batch_size,
                collate=collate,
                max_samples=max_samples
            )
    return activations


def compute_tsne(activations_list):
    combined_data = np.vstack([*activations_list])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random', n_iter=1000)
    tsne_results = tsne.fit_transform(combined_data)
    accumulated_lens = [0] + np.cumsum([len(x) for x in activations_list])
    
    tsne_activations_list = [
        tsne_results[accumulated_lens[i-1]:accumulated_lens[i]]
        for i in range(1, len(activations_list))
    ]
    return tsne_activations_list


def compute_pca(activations_list):
    combined_data = np.vstack([*activations_list])
    
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(combined_data)
    
    accumulated_lens = [0] + np.cumsum([len(x) for x in activations_list])
    
    pca_activations_list = [
        pca_results[accumulated_lens[i-1]:accumulated_lens[i]]
        for i in range(1, len(activations_list))
    ]
    return pca_activations_list


def plot_embeds_2d(embed_2d_list, keys, embed_type="t-SNE"):
    plt.figure(figsize=(10, 8))
    
    for i, embed_2d in enumerate(embed_2d_list):        
        plt.scatter(
            embed_2d[:, 0],
            embed_2d[:, 1],
            cmap='tab10',
            label=f"Activations {keys[i]}",
            alpha=0.7
        )
    
    plt.title(f"2D {embed_type} of Activations")
    plt.xlabel(f"{embed_type} Component 1")
    plt.ylabel(f"{embed_type} Component 2")
    plt.legend()
    plt.show()


def plot_activations(activations, dataset_names, embed_type="t-SNE"):
    activations_list = [
        values['activations'].cpu().numpy()
        for key, values in activations.items()
    ]
    if embed_type == "t-SNE":
        activations_2d_list = compute_tsne(activations_list)
    else:
        activations_2d_list = compute_pca(activations_list)

    plot_embeds_2d(activations_2d_list, keys=dataset_names, embed_type=embed_type)


def create_model(model_name="t5-base", tokenizer_name="T5Tokenizer", tokenizer_from_pretrained="google-t5/t5-base", max_length=256):
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(tokenizer_from_pretrained)

    model = T5forSummarization(**{
        "model_name": model_name,
        "cache_dir": "cache/",
        "max_length": max_length
    }).to('cuda:0')

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
        model_name=config["model_name"],
        tokenizer_name=config["tokenizer_name"],
        tokenizer_from_pretrained=config["tokenizer_from_pretrained"]
    )

    datasets_names = config["datasets_names"]
    mlm_datasets_indices = config["mlm_datasets_indices"]
    collates = [
        CollateClass(
            tokenizer=tokenizer,
            max_length=config["max_length"],
            mlm_items=mlm_items
        )
        for i, mlm_items in enumerate(mlm_datasets_indices)
    ]

    activations = collect_activations_by_datasets(
        model, tokenizer,
        datasets_configs=config["datasets"],
        embedding_specs=config["embedding_specs"],
        collates=collates,
        batch_size=config["batch_size"],
        max_samples=config["max_samples"]
    )
    plot_activations(activations, datasets_names, embed_type=config["embed_type_2d"])


if __name__ == "__main__":
    main()