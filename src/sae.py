import math
import warnings
from typing import Dict, Literal, Optional, Tuple

import hydra
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        architecture: Literal["relu", "jumprelu", "topk"] = "topk",
        input_dim: int = 768,
        feature_dim: int = 1536,
        topk: Optional[int] = None,
        apply_tied_weights: bool = False,
        apply_unit_norm: bool = True,
        apply_transpose_init: bool = True,
        apply_preprocessing: bool = True,
        init_scale: float = 1.0,
        encoder_bias: bool = True,
        decoder_bias: bool = False,
        preprocess_warmup_steps: int = 1000,
        ema_decay: float = 0.999,
        apply_auxiliary_loss: bool = True,
        k_aux: int = 512,
        alpha_aux: float = 1 / 32,
        feature_scope: Literal["token", "sample"] = "token",
        threshold_activation: float = -torch.inf,
        threshold_dead_steps: int = 10000,
        min_alive_neurons: int = 1,
        **kwargs,
    ):
        """
        Args:
            --- General ---
            architecture (Literal["relu", "jumprelu", "topk"]): Type of base architecture of SAE. Defaults to "topk"⭐.
            input_dim (int): Dimensionality of input activations. Defaults to 768.
            feature_dim (int): Dimensionality of sparse feature representation. Defaults to 1536 (= 2.0 of expension factor)⭐.

            --- Architecture-specific ---
            topk (Optional[int]): Number of top-k features to retain in 'topk' architecture. Required if architecture is 'topk'. Defaults to None (Refer to OpenAI) (Here we start from 64)⭐.

            --- Weights Initialization & Alignment ---
            apply_tied_weights (bool): Whether to use tied weights for encoder and decoder. Defaults to False (Refer to Cunningham & Anthropic).
            apply_unit_norm (bool): Whether to normalize decoder weights to unit norm. Recommended for 'relu' architecture. Defaults to True.
            apply_transpose_init (bool): Whether to initialize decoder weights as the transpose of encoder weights. Defaults to True (Refer to OpenAI).
            apply_preprocessing (bool): Whether to apply input preprocessing (centering and normalization). Defaults to True (Refer to Anthropic).
            init_scale (float): The gain for Xavier uniform initialization of encoder weights. Defaults to 1.0.
            encoder_bias (bool): Whether to include bias term in encoder. Defaults to True.
            decoder_bias (bool): Whether to include bias term in decoder. Defaults to False (Refer to OpenAI).
            preprocess_warmup_steps (int): Number of warm-up steps for input preprocessing. Defaults to 1000.
            ema_decay (float): Decay rate for Exponential Moving Average (EMA) in input preprocessing. Defaults to 0.999.

            --- Feature Utilization (Refer to OpenAI) ---
            apply_auxiliary_loss (bool): Whether to apply auxiliary loss to encourage dead features to activate. Defaults to True⭐.
            k_aux (int): Number of dead features to consider for auxiliary loss. Defaults to 512.
            alpha_aux (float): Coefficient for auxiliary loss term. Defaults to 1/32.
            feature_scope (Literal["token", "sample"]): If "token" selected, it means token-wise dead feature tracking (default). If "sample" selected, it means sample-wise dead feature tracking. Defaults to "sample".
            threshold_activation (float): Activation threshold to consider a feature as active. Defaults to -inf (i.e., any positive activation is considered active).
            threshold_dead_steps (int): Number of steps since last activation to consider a feature as dead. Defaults to 10000.
            min_alive_neurons (int): The minimum number of active neurons required for a feature to be considered 'alive' in a batch. Defaults to 1.
        """

        super().__init__()

        self.architecture = architecture
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.topk = topk

        self.apply_tied_weights = apply_tied_weights
        self.apply_unit_norm = apply_unit_norm
        self.apply_transpose_init = apply_transpose_init
        self.init_scale = init_scale
        self.encoder_bias = encoder_bias
        self.decoder_bias = decoder_bias
        self.apply_preprocessing = apply_preprocessing
        self.preprocess_warmup_steps = preprocess_warmup_steps
        self.ema_decay = ema_decay

        self.apply_auxiliary_loss = apply_auxiliary_loss
        self.k_aux = k_aux
        # self.alpha_aux = alpha_aux
        self.feature_scope = feature_scope
        self.threshold_activation = threshold_activation
        self.threshold_dead_steps = threshold_dead_steps
        self.min_alive_neurons = min_alive_neurons

        if self.architecture == "topk" and self.topk is None:
            raise ValueError("`topk` must be specified when architecture is 'topk'.")
        if self.topk > self.feature_dim:
            raise ValueError(f"`topk`({self.topk}) can not larger than `feature_dim`({self.feature_dim})!")

        self.encoder = nn.Linear(input_dim, self.feature_dim, bias=self.encoder_bias)

        if self.apply_tied_weights:
            self.decoder = TiedLinear(self.encoder, bias=self.decoder_bias)
        else:
            self.decoder = nn.Linear(self.feature_dim, self.input_dim, bias=self.decoder_bias)

        if self.apply_preprocessing:
            self.register_buffer("activation_mean", torch.zeros(self.input_dim))
            self.register_buffer("steps", torch.tensor(0))

        self._initialize_weights()

        if self.apply_auxiliary_loss:
            self.register_buffer("feature_counts", torch.zeros(self.feature_dim))
            self.register_buffer("steps_since_activation", torch.zeros(self.feature_dim))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forawrd pass of the SAE.

        Args:
            x: input activations of shape (batch_size, seq_len, input_dim)
            return_loss_components: if True, returns a dict of loss components for analysis

        Returns:
            features: sparse feature representation of shape (batch_size, seq_len, feature_dim)
            reconstructed: reconstructed activations of shape (batch_size, seq_len, input_dim)
            loss_components (Optional): dict containing `auxiliary loss`, `feature density`, `reconstruction loss`, etc.
        """

        # check input shape, 2-dim(token-wise) or 3-dim(batch-wise)
        flag_squeezed = x.dim() == 2
        if flag_squeezed:
            x = x.unsqueeze(0)

        processed_input = self._preprocess_input(x)
        encoded = self.encoder(processed_input)  # W_enc * x + b_enc

        features = self._apply_activation(encoded)  # sparse feature vector, ex) z = ReLU(W_enc * x + b_enc)

        reconstructed = self.decoder(features)  # reconstructed activation: x_hat = W_dec * z + b_dec

        if self.apply_preprocessing:
            reconstructed = reconstructed + self.activation_mean  # decentering of activation mean

        if self.apply_auxiliary_loss:
            aux_loss = self._compute_auxiliary_loss(x, features, reconstructed, encoded, attention_mask=attention_mask)

        # Normalize decoder weights for every steps during training
        if self.training and self.apply_unit_norm and not self.apply_tied_weights:
            self._normalize_decoder_weights()

        if flag_squeezed:
            features = features.squeeze(0)
            reconstructed = reconstructed.squeeze(0)

        loss_components = None
        if return_loss_components:
            recon_target = x.squeeze(0) if flag_squeezed else x
            recon_output = reconstructed.squeeze(0) if flag_squeezed else reconstructed

            if attention_mask is not None:
                recon_loss = F.mse_loss(recon_output[attention_mask == 1], recon_target[attention_mask == 1])
                feature_density = (features[attention_mask == 1] > 0).float().mean()
            else:
                recon_loss = F.mse_loss(recon_output, recon_target)
                feature_density = (features > 0).float().mean()

            num_dead_features = len(self._get_dead_features(step_threshold=self.threshold_dead_steps))

            loss_components = {
                "reconstruction_loss": recon_loss,
                "feature_density": feature_density,
                "dead_features": num_dead_features,
            }
            if self.architecture == "relu":
                if attention_mask is not None:
                    l1_penalty = features[attention_mask == 1].abs().mean()
                else:
                    l1_penalty = features.abs().mean()
                loss_components["l1_penalty"] = l1_penalty
            if self.apply_auxiliary_loss:
                loss_components["auxiliary_loss"] = aux_loss

        return features, reconstructed, loss_components

    def _initialize_weights(self):
        """Initialize weights of Encoder and Decoder, according to specified method."""

        # Encoder initialization with Xavier uniform
        nn.init.xavier_uniform_(self.encoder.weight, gain=self.init_scale)

        # Decoder initialization
        if self.apply_transpose_init:
            if self.apply_tied_weights:
                warnings.warn(
                    "Tied weights already enforce tranposition of decoder, `apply_transpose_init` has no effect because `apply_tied_weights` is True."
                )
            else:
                # decoder-weight initialized as the tranposition of encoder-weight: W_dec = W_enc ^ T
                with torch.no_grad():
                    self.decoder.weight.data = self.encoder.weight.data.t().clone()

        if self.apply_unit_norm:
            self._normalize_decoder_weights()

    def _normalize_decoder_weights(self):
        with torch.no_grad():
            if self.apply_tied_weights:
                # for tied weights, normalize encoder weights (which affects decoder)
                self.encoder.weight.data = F.normalize(self.encoder.weight.data, dim=1)
            else:
                # for untied weights, normalize decoder weights directly
                self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to input activations with EMA(Exponential Moving Average)."""

        if not self.apply_preprocessing:
            return x

        if self.training:
            with torch.no_grad():
                current_batch_mean = x.view(-1, self.input_dim).mean(dim=0).to(self.activation_mean)

                if self.steps < self.preprocess_warmup_steps:
                    # Warm-up phase: using accumulated mean
                    new_mean = (self.activation_mean * self.steps + current_batch_mean) / (self.steps + 1)
                else:
                    # EMA phase: using exponential moving average
                    new_mean = self.ema_decay * self.activation_mean + (1 - self.ema_decay) * current_batch_mean

                self.activation_mean.copy_(new_mean)
                self.steps += 1

        processed = x - self.activation_mean

        # activation normalization (fit each vector's norm ~ sqrt(input_dim))
        current_norm = processed.norm(dim=-1, keepdim=True).detach()
        target_norm = math.sqrt(self.input_dim)

        processed = processed * (target_norm / (current_norm + 1e-8))

        return processed

    def _apply_activation(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.architecture == "relu":
            return self._relu_activation(encoded)

        elif self.architecture == "topk":
            return self._topk_activation(encoded)

        elif self.architecture == "jumprelu":
            return self._jumprelu_activation(encoded)

        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

    def _relu_activation(self, encoded: torch.Tensor) -> torch.Tensor:
        return F.relu(encoded)

    def _topk_activation(self, encoded: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = encoded.shape
        encoded_flat = encoded.view(-1, feature_dim)

        topk_values, topk_indices = torch.topk(encoded_flat, self.topk, dim=-1)

        features = torch.zeros_like(encoded_flat)
        features.scatter_(-1, topk_indices, topk_values)

        return features.view(batch_size, seq_len, feature_dim)

    def _jumprelu_activation(self, encoded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("JumpReLU activation is not implemented yet.")

    def _compute_auxiliary_loss(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        reconstructed: torch.Tensor,
        encoded: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes auxiliary loss to to encourage dead features to participate in reconstruction."""

        if not self.apply_auxiliary_loss or not self.training:
            return torch.tensor(0.0, device=x.device)

        with torch.no_grad():
            if attention_mask is not None:
                masked_features = features * attention_mask.unsqueeze(-1)
            else:
                masked_features = features

            if self.feature_scope == "token":
                # token-wise dead features tracking (default): if a feature is active in any token of the batch, it is considered active
                active_features = (masked_features > self.threshold_activation).float().sum(dim=[0, 1])

            elif self.feature_scope == "sample":
                # sample-wise dead features tracking: if a feature is active in any token of a sample, it is considered active
                active_features = (masked_features > self.threshold_activation).any(dim=1).float().sum(dim=0)

            self.feature_counts += active_features
            self.steps_since_activation += 1
            active_mask = active_features >= self.min_alive_neurons
            self.steps_since_activation[active_mask] = 0

        dead_features = self.steps_since_activation > self.threshold_dead_steps

        if not dead_features.any():
            return torch.tensor(0.0, device=x.device)

        dead_indices = torch.where(dead_features)[0]

        if len(dead_indices) > self.k_aux:
            perm = torch.randperm(len(dead_indices), device=dead_indices.device)
            sampled_indices = dead_indices[perm[: self.k_aux]]
        else:
            sampled_indices = dead_indices

        with torch.no_grad():
            reconstruction_error = x - reconstructed
            if attention_mask is not None:
                reconstruction_error = reconstruction_error * attention_mask.unsqueeze(-1)
            target = reconstruction_error.detach()

        # reconstruction error using only dead features
        dead_encoded = encoded[..., sampled_indices]
        dead_features_act = F.relu(dead_encoded)  # NOTE: using ReLU to activate dead features
        dead_decoder_weights = self.decoder.weight[:, sampled_indices]
        dead_reconstruction = torch.matmul(dead_features_act, dead_decoder_weights.t())

        # L2 loss on the reconstruction (MSE)
        if attention_mask is not None:
            aux_loss = F.mse_loss(dead_reconstruction[attention_mask == 1], target[attention_mask == 1])
        else:
            aux_loss = F.mse_loss(dead_reconstruction, target)

        return aux_loss

    def _get_dead_features(self, step_threshold: int) -> torch.Tensor:
        if hasattr(self, "steps_since_activation"):
            return torch.where(self.steps_since_activation > step_threshold)[0]
        return torch.tensor([], dtype=torch.long, device=self.encoder.weight.device)


class TiedLinear(nn.Module):
    def __init__(self, tied_layer: nn.Linear, bias: bool = False):
        super().__init__()
        self.tied_layer = tied_layer
        if bias:
            self.bias = nn.Parameter(torch.zeros(tied_layer.in_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using tied weights.

        Args:
            x: input tensor of shape (batch_size, seq_len, feature_dim)
            # encoder.weight shape: (feature_dim, input_dim)
            # encoder.weight.t() shape: (input_dim, feature_dim)

        Returns:
            output: tensor of shape (batch_size, seq_len, input_dim)
            # output = x @ encoder.weight.t() + bias
        """

        weight = self.tied_layer.weight.t()
        return F.linear(x, weight, self.bias)


class SAELoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        l1_weight: float = 1e-3,
        auxiliary_weight: float = 1.0,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.l1_weight = l1_weight
        self.auxiliary_weight = auxiliary_weight

    def forward(
        self,
        loss_components: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute SAE loss from loss components (dictionary of pre-computed loss components).

        Args:
            loss_components: Dictionary from SAE model, containing at least 'reconstruction_loss'. Can also contain 'l1_penalty' and 'auxiliary_loss'

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of all individual and weighted loss components
        """

        loss_dict = loss_components.copy()

        # 1. Reconstruction loss (by default, MSE)
        total_loss = self.reconstruction_weight * loss_dict["reconstruction_loss"]
        loss_dict["reconstruction_loss_weighted"] = total_loss.detach()

        # 2. L1 sparsity penalty, if available (ReLU)
        if "l1_penalty" in loss_dict:
            l1_loss_weighted = self.l1_weight * loss_dict["l1_penalty"]
            total_loss += l1_loss_weighted
            loss_dict["l1_loss_weighted"] = l1_loss_weighted.detach()

        # 3. Auxiliary loss, if available (TopK)
        if "auxiliary_loss" in loss_dict:
            if loss_dict["auxiliary_loss"] > 0:
                aux_loss_weighted = self.auxiliary_weight * loss_dict["auxiliary_loss"]
                total_loss += aux_loss_weighted
                loss_dict["auxiliary_loss_weighted"] = aux_loss_weighted.detach()

        return total_loss, loss_dict


class SAELightningModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
    ):
        super().__init__()
        self.model: SparseAutoencoder = hydra.utils.instantiate(model_cfg)
        self.loss_fn: SAELoss = hydra.utils.instantiate(loss_cfg)
        self.optimizer_cfg = optimizer_cfg

        self.save_hyperparameters()

    def _shared_step(self, batch, batch_idx):
        activations, attention_mask, _indices = batch

        _, _, loss_components = self.model(
            activations,
            attention_mask=attention_mask,
            return_loss_components=True,
        )
        total_loss, loss_dict = self.loss_fn(loss_components)
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        total_loss, loss_dict = self._shared_step(batch, batch_idx)

        log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, loss_dict = self._shared_step(batch, batch_idx)

        log_dict = {f"val/{k}": v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
        return optimizer
