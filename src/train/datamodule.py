from pathlib import Path
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.clearml_pipeline.preprocess.task import get_prep_data
from src.config import MLPExperimentConfig, RunModeEnum
from src.data.preprocessing.main import download_csv, preprocess_data
from src.train.dataset import TabularDataset


class TabularDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: MLPExperimentConfig,
    ):
        super().__init__()
        self.project_name = cfg.project_name
        self.cfg = cfg
        self.data_cfg = cfg.data_config
        self.prep_cfg = cfg.data_config.processing_config

        self.batch_size = cfg.dataloader_config.batch_size
        self.num_workers = cfg.dataloader_config.num_workers
        self.pin_memory = cfg.dataloader_config.pin_memory

        # There is no need to download and read datasets on each prepare_data() and setup() hooks call
        self.is_data_prepared: bool = False
        self.is_fit_set_up: bool = False
        self.is_test_set_up: bool = False

        self.data_train: Optional[TabularDataset] = None
        self.data_val: Optional[TabularDataset] = None
        self.data_test: Optional[TabularDataset] = None

        # Prevent hyperparameters from being stored in checkpoints.
        self.save_hyperparameters(logger=False)

    def _prep_data_attrs(self) -> None:
        self.prepare_data()
        self.setup(stage='test')

    @property
    def num_classes(self) -> int:
        self._prep_data_attrs()
        return self.data_test.num_classes  # type: ignore

    @property
    def num_features(self) -> int:
        self._prep_data_attrs()
        return self.data_test.num_features  # type: ignore

    def prepare_data(self) -> None:
        if self.is_data_prepared:
            return

        # Since we don't do distributed training, we can safely assign state here
        self.data_path = self._prepare_data()

        self.is_data_prepared = True

    def _prepare_data(self) -> Path:
        if self.cfg.run_mode == RunModeEnum.pipeline:
            return get_prep_data(self.cfg)
        elif self.cfg.run_mode == RunModeEnum.local:
            raw_csv_path = download_csv(self.project_name, self.data_cfg, skip_if_exists=True)
            return preprocess_data(self.project_name, self.data_cfg, raw_csv_path, seed=self.cfg.seed)

    def setup(self, stage: str) -> None:
        if stage == 'fit' and not self.is_fit_set_up:
            self.data_train = TabularDataset(self.data_path, 'train', self.prep_cfg.target_column)
            self.data_val = TabularDataset(self.data_path, 'val', self.prep_cfg.target_column)
            self.is_fit_set_up = True

        elif stage == 'test' and not self.is_test_set_up:
            self.data_test = TabularDataset(self.data_path, 'test', self.prep_cfg.target_column)
            self.is_test_set_up = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


def dm_prepare_data(cfg: MLPExperimentConfig) -> None:
    datamodule = TabularDataModule(cfg)
    datamodule.prepare_data()
