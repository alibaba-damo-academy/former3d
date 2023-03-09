import argparse
import glob
import os
import subprocess
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from former3d import data, lightningmodel


class CudaClearCacheCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # if training needs to be resumed from checkpoint,
    # it is helpful to change the seed so that
    # the same data augmentations are not re-used
    pl.seed_everything(config["seed"])
    save_dir = os.path.join("results", config["wandb_project_name"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], save_dir=save_dir, config=config, offline=False) # Note offline
    
    if os.getenv("LOCAL_RANK", '0') == '0':
        subprocess.call(
            [
                "zip",
                "-q",
                os.path.join(logger.experiment.dir, "code.zip"),
                *glob.glob("./config*"),
                *glob.glob("model/*.py"),
                *glob.glob("model/net3d/*.py"),
                *glob.glob("scripts/*.py"),
            ]
        )
        ckpt_dir = os.path.join(logger.experiment.dir, "ckpts")
    else:
        ckpt_dir = os.path.join(config["wandb_project_name"], "ckpts")
    checkpointer = pl.callbacks.ModelCheckpoint(
        save_last=True,
        dirpath=ckpt_dir,
        verbose=True,
        filename='{epoch:03d}-{val/voxel_loss_fine:.3f}', auto_insert_metric_name=False,
        save_top_k=-1,
        monitor="val/loss",
    )
    callbacks = [checkpointer, lightningmodel.FineTuning(config["initial_epochs"]), CudaClearCacheCallback()]
    
    if config["use_amp"]:
        amp_kwargs = {"precision": 16}
    else:
        amp_kwargs = {}
    
    model = lightningmodel.LightningModel(config)
    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy='ddp',
        # sync_batchnorm=True,
        # plugins=DDPPlugin(find_unused_parameters=False),
        num_sanity_val_steps=0,
        logger=logger,
        benchmark=False,
        max_epochs=config["initial_epochs"] + config["finetune_epochs"],
        check_val_every_n_epoch=10,
        detect_anomaly=False,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=10,
        **amp_kwargs,
    )
    trainer.fit(model, ckpt_path=config["ckpt"])
