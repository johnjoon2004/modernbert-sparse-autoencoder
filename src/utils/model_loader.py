import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import OmegaConf
from src.sae import SparseAutoencoder
from src.global_settings import settings

PROJECT_ROOT = settings.project_root


@dataclass
class SAELoadingConfig:
    trained_model_path: Path | str | None = None
    model_tag: str | None = None
    hparamset_tag: str | None = None
    layer_idx: int | None = None
    return_ckpt: bool = False


def _get_config_file_path(model_dir: Path) -> Path:
    default_config_filename = "config.yaml"
    config_files = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
    if not config_files:
        raise FileNotFoundError(f"No yaml files found in directory: {model_dir}")
    if len(config_files) == 1:
        config_path = config_files[0]
    else:
        default_config_path = model_dir / default_config_filename
        if default_config_path.exists():
            config_path = default_config_path
        else:
            config_path = config_files[0]
            warnings.warn(f"Multiple yaml files found in {model_dir}. Using {config_path.name}.")
    return config_path


def _get_weights_file_path(model_dir: Path) -> Path:
    highest_priority_filename = "final_model.pt"
    weight_files = list(model_dir.glob("*.pt"))
    if not weight_files:
        raise FileNotFoundError(f"No .pt files found in directory: {model_dir}")
    if len(weight_files) == 1:
        weights_path = weight_files[0]
    else:
        final_model_dir = model_dir / highest_priority_filename
        if final_model_dir.exists():
            weights_path = final_model_dir
        else:
            weights_path = weight_files[0]
            warnings.warn(f"Multiple .pt files found in {model_dir}. Using {weights_path.name}.")
    return weights_path


def load_sae(
    checkpoint: Optional[str] = None,
    trained_model_path: Optional[str | Path] = None,
    model_tag: Optional[str] = None,
    hparamset_tag: Optional[str] = None,
    layer_idx: Optional[int] = None,
    return_ckpt: bool = False,
) -> SparseAutoencoder | tuple[SparseAutoencoder, str]:
    """
    Factory function to load a trained Sparse Autoencoder model based on the provided config file path and weights path.

    Args:
        trained_model_path (str | Path): The path to the model weights file / directory.
          If provided, model_tag, hparamset_tag, and layer_idx are ignored. (vice versa)
        model_tag (str): The model identifier tag.
        hparamset_tag (str): The hyperparameter set identifier tag.
        layer_idx (int): The layer index.
        return_ckpt (bool): Whether to return the run_tag along with the model.

    Returns:
        SparseAutoencoder: The loaded Sparse Autoencoder model.
    """

    if trained_model_path is not None:
        trained_model_path = Path(trained_model_path)
        if not trained_model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {trained_model_path}")

        if trained_model_path.is_dir():
            # directory -> find weights file(.pt) and config file(.yaml)
            weights_file = _get_weights_file_path(trained_model_path)
            config_file = _get_config_file_path(trained_model_path)
            run_tag = trained_model_path.name
        elif trained_model_path.is_file() and trained_model_path.suffix == ".pt":
            weights_file = trained_model_path
            # path of a '.pt' file
            # -> first find if config yaml exists in the same directory
            # -> if not, find config in ../config/ using **model_tag**
            #   - try find with provided model_tag.
            #   - if not provided, try parse `model_tag` from `run_tag` from parent directory
            #   - finally find config file in '~/conf/model/{model_tag}.yaml'
            parent_dir = trained_model_path.parent
            run_tag = parent_dir.name
            try:
                config_file = _get_config_file_path(parent_dir)
            except FileNotFoundError:
                if model_tag is None:
                    if "_" not in run_tag:
                        raise ValueError(f"Cannot parse model_tag from run_tag: {run_tag}")
                    model_tag = "_".join(run_tag.split("_")[:2])  # assumed model_tag format: f{feat_dim}_k{topk}
                config_file = PROJECT_ROOT / "src" / "conf" / "model" / f"{model_tag}.yaml"
        else:
            raise ValueError(f"Invalid trained_model_path: {trained_model_path}")
    else:
        if checkpoint is not None:
            try:
                _model_tag, _topk, hparamset_tag = checkpoint.split("_")[:3]
                model_tag = _model_tag + "_" + _topk
            except ValueError:
                raise ValueError(
                    f"Invalid checkpoint format: {checkpoint}. Expected format: {{model_tag}}_{{hparamset_tag}}"
                )

        if None in (model_tag, hparamset_tag, layer_idx):
            raise ValueError(
                "If 'trained_model_path' is not provided, all of 'model_tag', 'hparamset_tag', and 'layer_idx' must be provided."
            )
        run_tag = f"{model_tag}_{hparamset_tag}_L{layer_idx}"
        config_file = PROJECT_ROOT / "src" / "conf" / "model" / f"{model_tag}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Model config file not found: {config_file}")
        weights_dir = PROJECT_ROOT / "outputs" / "weights" / run_tag
        if not weights_dir.exists() or not weights_dir.is_dir():
            raise FileNotFoundError(f"Model weights directory not found: {weights_dir}")
        weights_file = _get_weights_file_path(weights_dir)

    model_args = OmegaConf.load(config_file).args
    model: SparseAutoencoder = hydra.utils.instantiate(model_args)
    state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    if return_ckpt:
        return model, run_tag

    return model
