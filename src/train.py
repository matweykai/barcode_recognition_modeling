import os
import clearml
from os import path as osp

import click

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar, ModelCheckpoint
from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import OCRDM
from src.lightning_module import OCRModule


@click.command()
@click.argument('config_path')
def main(config_path: str):
    config = Config.from_yaml(config_path)
    pl.seed_everything(config.rand_seed, workers=True)
    train(config)


def train(config: Config):
    datamodule = OCRDM(config.data_config)
    model = OCRModule(config)

    experiment_save_path = osp.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    task = clearml.Task.init(
        project_name=config.project_name,
        task_name=config.experiment_name,
        auto_connect_frameworks=True,
    )

    task.connect(config.model_dump())

    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            RichProgressBar(),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == '__main__':
    main()
