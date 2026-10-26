import argparse

import torch
from src.activations import ActivationDataset
from src.global_settings import settings
from src.sae_feature_handler import SaeFeatureHandler
from src.utils.model_loader import load_sae
from torch.utils.data import DataLoader

LM_WEIGHTS_PATH = settings.lm_weights_path
TEXT_DATASET_PATH = settings.text_dataset_path

ACTIVATION_DATASET_DIR_TEMPLATE = (
    str(settings.rs_data_dir / "layer_{layer_idx}")
)
BS = 128
DTYPE = torch.float32

DEFAULT_LAYERS = settings.layers


def init_dataloader(layer_index) -> DataLoader:
    dataset = ActivationDataset(data_dir=ACTIVATION_DATASET_DIR_TEMPLATE.format(layer_idx=layer_index), dtype=DTYPE)
    dataloader = DataLoader(dataset, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader


def init_handler(model_tag, hparamset_tag, layer_index, feature_indices, dataloader) -> SaeFeatureHandler:
    model = load_sae(
        trained_model_path=None,
        model_tag=model_tag,
        hparamset_tag=hparamset_tag,
        layer_idx=layer_index,
        return_ckpt=False,
    )
    checkpoint = f"{model_tag}_{hparamset_tag}"
    handler = SaeFeatureHandler(
        model=model,
        checkpoint=checkpoint,
        layer=layer_index,
        feature_indices=feature_indices,
        dataloader=dataloader,
    )
    return handler


DEFAULT_TOP_N = 50


def main(args):
    model_tag = args.model
    hparamset_tag = args.hparamset
    layer_indices = args.layers
    feature_indices = args.features
    top_n = args.top_n

    for layer in layer_indices:
        dataloader = init_dataloader(layer)
        handler = init_handler(model_tag, hparamset_tag, layer, feature_indices, dataloader)

        top_pos_indices, top_neg_indices = handler.process_all_features(top_n)
        handler.save_topn_records(top_pos_indices, top_neg_indices)


if __name__ == "__main__":
    """
    Usage Examples:
        python -m scripts.extract_feature_records --model f3072_k256
    """

    parser = argparse.ArgumentParser(
        description="Reconstruct text samples' activations per feature using trained SAE model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Tag name for the trained SAE model (e.g., 'f960_k8')",
    )

    parser.add_argument(
        "--hparamset",
        type=str,
        default="alpha",
        help="Hyperparameter to vary (e.g., 'alpha', 'beta')",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="List of layer indices to decode features from. (e.g., --layers 1 6 11 15)",
    )

    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=None,
        help="List of feature indices to decode. (default: all features)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top activations to record per feature.",
    )

    args = parser.parse_args()

    main(args)
