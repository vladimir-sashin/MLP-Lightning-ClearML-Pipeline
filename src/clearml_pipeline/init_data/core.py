from pathlib import Path
from typing import Dict

from clearml import Dataset
from clearml.task import TaskInstance

from src.clearml_pipeline.utils import DataManager
from src.config import DataConfig


class RawDataManager(DataManager):
    def upload_raw_ds(self, local_csv_path: Path) -> Dataset:
        dataset = Dataset.create(
            dataset_project=self.project_name,
            dataset_name=self.task_dataset_name,  # works only if use_current_task=False
            description=self.data_cfg.dataset_description,
            use_current_task=True,
        )
        dataset.add_files(local_csv_path, verbose=True)
        dataset.tags = ['raw']
        dataset.finalize(auto_upload=True)

        self.logger.report_text(
            f'Raw `{self.task_dataset_name}` dataset has been created in ClearML in the `{self.project_name}` '
            f'project!',
        )
        return dataset


def connect_cfg(task: TaskInstance, data_cfg: DataConfig) -> DataConfig:
    tracked_cfg_dump = get_init_cfg_dump(data_cfg)

    task.connect(tracked_cfg_dump)

    data_cfg = data_cfg.model_copy(deep=True)
    for key, val in tracked_cfg_dump.items():
        setattr(data_cfg, key, val)

    return data_cfg


def get_init_cfg_dump(data_cfg: DataConfig) -> Dict[str, str]:
    return data_cfg.model_dump(include=CFG_KEYS_TO_TRACK)  # type: ignore[no-any-return]


CFG_KEYS_TO_TRACK = {'orig_dataset_name', 'raw_csv_url', 'dataset_description'}
