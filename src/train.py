import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.config import MLPExperimentConfig, get_experiment_cfg
from src.constants import MLP_CFG_PATH
from src.datamodule import TabularDataModule
from src.lightning_module import ClassificationLightningModule


def train_mlp(cfg: MLPExperimentConfig) -> None:
    lightning.seed_everything(cfg.seed)

    datamodule = TabularDataModule(cfg=cfg.data_config)
    model = ClassificationLightningModule(cfg, datamodule.num_features, datamodule.num_classes)

    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor='valid_f1', mode='max', every_n_epochs=1)
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        checkpoint_callback,
    ]

    trainer = Trainer(**dict(cfg.trainer_config), callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfg = get_experiment_cfg(MLP_CFG_PATH)
    train_mlp(cfg)
