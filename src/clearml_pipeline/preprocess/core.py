from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from clearml import Dataset
from clearml.task import TaskInstance

from src.clearml_pipeline.utils import DataManager, get_data_task_name
from src.config import DataConfig, MLPExperimentConfig
from src.data.preprocessing.main import RAW_CSV_FILENAME


def connect_cfg(task: TaskInstance, cfg: MLPExperimentConfig) -> MLPExperimentConfig:
    tracked_cfg_dump = get_prep_cfg_dump(cfg)

    task.connect(tracked_cfg_dump)

    cfg = cfg.model_copy(deep=True)
    cfg.seed = tracked_cfg_dump.pop('seed')
    cfg.data_config.orig_dataset_name = tracked_cfg_dump.pop('orig_dataset_name')
    for key, val in tracked_cfg_dump.items():
        setattr(cfg.data_config.processing_config, key, val)
    return cfg


class PreprocessDataManager(DataManager):
    def __init__(self, project_name: str, task_dataset_name: str, data_cfg: DataConfig, logger: Logger):
        super().__init__(project_name, task_dataset_name, data_cfg, logger)
        self.raw_dataset: Dataset = self._check_raw_ds_exists()

    def _check_raw_ds_exists(self) -> Dataset:
        raw_ds_name = get_data_task_name(
            self.data_cfg,
            stage='init',
        )
        if (raw_dataset := self.get_ds_if_exists(dataset_name=raw_ds_name, alias='raw_dataset')) is None:
            raise ValueError
        return raw_dataset

    def get_latest_preprocessed_ds(self) -> Optional[Dataset]:
        return self.get_ds_if_exists(dataset_name=self.task_dataset_name, alias='latest_preprocessed_dataset')

    def upload_processed_ds(self, processed_dir: Path) -> Dataset:
        latest_processed_ds = self.get_latest_preprocessed_ds()
        if latest_processed_ds is None:
            return self._upload_first_version(processed_dir)
        elif latest_processed_ds.verify_dataset_hash(str(processed_dir)):
            return self._upload_new_version(processed_dir, latest_processed_ds)
        else:
            self.logger.report_text(
                f'Pre-processed dataset is equal to the latest version of `{self.task_dataset_name}` dataset in the '
                f'`{self.project_name}` project. No upload needed, using the latest version.',
            )
            return latest_processed_ds

    def _upload_processed_ds(
        self,
        processed_dir: Path,
        parent_ds: Dataset,
        tags: Sequence[str] = ('preprocessed',),
    ) -> Dataset:
        processed_ds = Dataset.create(
            use_current_task=True,
            dataset_project=self.project_name,
            dataset_name=self.task_dataset_name,  # works only if use_current_task=False
            parent_datasets=[parent_ds],
            dataset_tags=tags,
        )
        processed_ds.sync_folder(local_path=processed_dir, verbose=True)
        processed_ds.finalize(auto_upload=True)
        return processed_ds

    def _upload_first_version(self, processed_dir: Path) -> Dataset:
        processed_ds = self._upload_processed_ds(processed_dir, self.raw_dataset)
        self.logger.report_text(
            f'The first version of the `{self.task_dataset_name}` dataset has been created in ClearML in the '
            f'`{self.project_name}` project!',
        )
        return processed_ds

    def _upload_new_version(self, processed_dir: Path, latest_processed_ds: Dataset) -> Dataset:
        processed_ds = self._upload_processed_ds(processed_dir, latest_processed_ds)
        self.logger.report_text(
            f'A new version of the `{self.task_dataset_name}` dataset has been created in ClearML in the '
            f'`{self.project_name}` project!',
        )
        return processed_ds


def get_raw_ds_local_path(raw_dataset: Dataset) -> Path:
    return Path(raw_dataset.get_local_copy()) / RAW_CSV_FILENAME


def get_prep_cfg_dump(cfg: MLPExperimentConfig) -> Dict[str, Any]:
    data_cfg = cfg.data_config
    tracked_cfg_dump = data_cfg.processing_config.model_dump()
    tracked_cfg_dump['seed'] = cfg.seed
    tracked_cfg_dump['orig_dataset_name'] = data_cfg.orig_dataset_name
    return tracked_cfg_dump  # type: ignore[no-any-return]
