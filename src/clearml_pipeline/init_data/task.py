from clearml import Dataset

from src.clearml_pipeline.init_data.core import RawDataManager, connect_cfg
from src.clearml_pipeline.utils import get_data_task_name, init_task
from src.config import MLPExperimentConfig, get_experiment_cfg
from src.data.preprocessing.main import download_csv


def clearml_init_data(experiment_cfg: MLPExperimentConfig, create_draft: bool = False) -> Dataset:
    data_cfg = experiment_cfg.data_config
    project_name = experiment_cfg.project_name

    task_dataset_name = get_data_task_name(data_cfg, stage='init')
    task, logger = init_task(project_name, task_dataset_name)
    if create_draft is True:
        task.execute_remotely()

    logger.report_text(f'Initializing raw `{task_dataset_name}` dataset in ClearML, project {project_name}...')

    data_cfg = connect_cfg(task, data_cfg)
    data_manager = RawDataManager(
        task_dataset_name=task_dataset_name,
        logger=logger,
        project_name=project_name,
        data_cfg=data_cfg,
    )

    if existing_dataset := data_manager.get_ds_if_exists(alias='raw_dataset'):
        return existing_dataset
    local_csv_path = download_csv(project_name, data_cfg, skip_if_exists=True)
    ds = data_manager.upload_raw_ds(local_csv_path)
    return ds


if __name__ == '__main__':
    cfg = get_experiment_cfg()
    clearml_init_data(cfg, True)
