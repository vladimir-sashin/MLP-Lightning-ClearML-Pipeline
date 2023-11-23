from pathlib import Path
from typing import Optional

import invoke
from lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader

from src import heart_data
from src.config import DataConfig
from src.constants import PROJECT_ROOT
from src.dataset import TabularDataset
from src.preprocessing import preprocess_data


class TabularDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.data_path: Path = PROJECT_ROOT / self.cfg.processed_path

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
        # TODO: parametrize data downloading to enable training on custom datasets, not only Heart Disease Dataset
        heart_data.download(invoke.Context(), self.cfg)
        preprocess_data(self.cfg)

        self.is_data_prepared = True

    def setup(self, stage: str) -> None:
        if stage == 'fit' and not self.is_fit_set_up:
            self.data_train = TabularDataset(self.data_path, 'train', self.cfg.target_column)
            self.data_val = TabularDataset(self.data_path, 'val', self.cfg.target_column)
            self.is_fit_set_up = True

        elif stage == 'test' and not self.is_test_set_up:
            self.data_test = TabularDataset(self.data_path, 'test', self.cfg.target_column)
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


def dm_prepare_data(cfg: DataConfig, seed: int) -> None:
    seed_everything(seed)
    datamodule = TabularDataModule(cfg)
    datamodule.prepare_data()
