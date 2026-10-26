from contextlib import contextmanager
from typing import Any, Callable, Dict, Literal, Tuple

import torch
import torch.nn as nn
from src.sae import SparseAutoencoder
from transformers import BatchEncoding, PreTrainedModel


class SAEAdapter:
    """
    SAE intervention model for feature-steering on transformer's particular layer.

    Using `register_forward_pre_hook` to intervene the input hidden states (residual stream) before layer-norm and attention,
    reconstructed from the SAE decoded feature vector and added to the original activation.
    """

    def __init__(self, model: PreTrainedModel, sae: SparseAutoencoder):
        self.model = model
        self.sae = sae
        self.sae.eval()
        self._hook_handle = None

        self.results: Dict[str, Any] = {}

    def _get_target_module(self, layer: int) -> nn.Module:
        """
        Select one layer-module to intervene.
        Currently only implemented specific transformer architecture, such as ModernBERT.
        """

        # TODO: Extend to more transformer architectures

        try:
            return self.model.model.layers[layer]  # ModernBERT
        except AttributeError:
            try:
                return self.model.bert.encoder.layer[layer]  # BERT
            except Exception as e:
                return ValueError(
                    f"Unsupported model architecture for intervention. (layer: {layer})"
                    f"Please implement the `_get_target_module` method according to this model."
                    f"Error: {e}"
                )

    def _get_steering_hook(
        self,
        feature_index: int,
        mode: Literal["activate", "deactivate"],
        strength: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> Callable:

        def hook(module, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
            # inputs[0] equals the residual stream input to the layer (hidden states)
            # (batch_size, seq_len, hidden_dim)
            activation = inputs[0].to(device)

            self.sae.to(device)

            with torch.no_grad():
                # (1) Feature Extraction (Encoding)
                processed_input = self.sae._preprocess_input(activation)
                encoded = self.sae.encoder(processed_input)
                features_org = self.sae._apply_activation(encoded)

                feature_score_org = features_org[..., feature_index].sum().item()
                self.results["feature_score_org"] = feature_score_org

                # (2) Feature Steering
                features_rec = features_org.clone()
                if mode == "activate":
                    features_rec[..., feature_index] = strength
                elif mode == "deactivate":
                    features_rec[..., feature_index] = 0.0

                feature_score_rec = features_rec[..., feature_index].sum().item()
                self.results["feature_score_rec"] = feature_score_rec

                # (3) Reconstruction (Decoding)
                activation_rec = self.sae.decoder(features_rec)
                if self.sae.apply_preprocessing:
                    activation_rec = activation_rec + self.sae.activation_mean.to(device)  # decentering

            # Replace activation with reconstructed(steered) version
            activation_rec = activation_rec.to(activation.device)

            inputs_ = (activation_rec,) + inputs[1:]  # keep other inputs unchanged (attention_mask, etc.)
            return inputs_

        return hook

    @contextmanager
    def steer(
        self,
        inputs: Dict[str, torch.Tensor] | BatchEncoding,
        layer: int,
        feature_index: int,
        mode: Literal["activate", "deactivate"],
        strength: float = 1.0,
    ):
        self.results.clear()

        target_module = self._get_target_module(layer=layer)
        device = next(self.model.parameters()).device

        _inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            # Compute original outputs (no steering)
            # ------------------ BEFORE -----------------------
            with torch.no_grad():
                outputs_org = self.model(**_inputs)
                logits_org = outputs_org.logits.detach().cpu()
                self.results["logits_org"] = logits_org
            # -------------------------------------------------

            # Register Steering Hook
            hook_fn = self._get_steering_hook(
                feature_index=feature_index,
                mode=mode,
                strength=strength,
                device=device,
            )
            self._hook_handle = target_module.register_forward_pre_hook(
                hook_fn
            )  # before `forward` of the target module

            # Recompute outputs with intervention (steered)
            # ------------------ AFTER -----------------------
            with torch.no_grad():
                outputs_rec = self.model(**_inputs)
                logits_rec = outputs_rec.logits.detach().cpu()
                self.results["logits_rec"] = logits_rec
            # -------------------------------------------------

            yield self.results

        finally:
            if self._hook_handle:
                self._hook_handle.remove()
                self._hook_handle = None
