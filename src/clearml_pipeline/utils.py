from typing import Literal, Optional

from clearml import Dataset, Logger, Task, TaskTypes
from clearml.task import TaskInstance

from src.config import DataConfig


def get_data_task_name(data_cfg: DataConfig, stage: Literal['init', 'prep']) -> str:
    prefix = stage[0].upper() + stage[1:]
    return f'{prefix} {data_cfg.orig_dataset_name}'


def init_task(project_name: str, task_dataset_name: str) -> tuple[TaskInstance, Logger]:
    Task.force_requirements_env_freeze()
    task = Task.init(
        project_name=project_name,
        task_name=task_dataset_name,
        task_type=TaskTypes.data_processing,
        reuse_last_task_id=False,
    )

    return task, task.get_logger()


class DataManager:
    def __init__(self, project_name: str, task_dataset_name: str, data_cfg: DataConfig, logger: Logger):
        self.project_name = project_name
        self.task_dataset_name = task_dataset_name
        self.data_cfg = data_cfg
        self.logger = logger

    def get_ds_if_exists(self, dataset_name: Optional[str] = None, alias: str = 'dataset') -> Optional[Dataset]:
        if dataset_name is None:
            dataset_name = self.task_dataset_name
        existing_ds_names = {ds['name'] for ds in Dataset.list_datasets(dataset_project=self.project_name)}
        if dataset_name in existing_ds_names:
            self.logger.report_text(f'`{dataset_name}` dataset already exists in `{self.project_name}` project.')
            return Dataset.get(
                dataset_project=self.project_name,
                dataset_name=dataset_name,
                alias=alias,
            )
        return None
