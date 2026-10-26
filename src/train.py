import os

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from src.sae import SAELightningModule

verbose = False


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    if verbose:
        print("===== Configuration =====")
        print(OmegaConf.to_yaml(cfg))
        print("=========================")

    torch.set_float32_matmul_precision("high")

    pl.seed_everything(cfg.training.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    model = SAELightningModule(
        model_cfg=cfg.model.args,
        loss_cfg=cfg.model.loss,
        optimizer_cfg=cfg.model.optimizer,
    )

    loggers = [hydra.utils.instantiate(lg_cfg) for lg_cfg in cfg.training.trainer.logger]
    callbacks = [hydra.utils.instantiate(cb_cfg) for cb_cfg in cfg.training.trainer.callbacks]

    trainer = hydra.utils.instantiate(cfg.training.trainer, logger=loggers, callbacks=callbacks)

    trainer.fit(model, datamodule)

    if hasattr(cfg.training, "final_weights_path"):
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            os.makedirs(os.path.dirname(cfg.training.final_weights_path), exist_ok=True)
            final_model_state_dict = SAELightningModule.load_from_checkpoint(best_model_path).model.state_dict()
            torch.save(final_model_state_dict, cfg.training.final_weights_path)
            print(f"Final model weights saved to: {cfg.training.final_weights_path}")

    print(" ✅ Finished training!")


if __name__ == "__main__":
    train()
