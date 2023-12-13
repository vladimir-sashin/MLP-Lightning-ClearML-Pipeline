from pathlib import Path

from clearml import Dataset

from src.clearml_pipeline.preprocess.core import PreprocessDataManager, connect_cfg, get_raw_ds_local_path
from src.clearml_pipeline.utils import get_data_task_name, init_task
from src.config import MLPExperimentConfig, get_experiment_cfg
from src.data.preprocessing.main import preprocess_data


def clearml_preprocess(cfg: MLPExperimentConfig, create_draft: bool = False) -> Dataset:
    project_name = cfg.project_name

    task_dataset_name = get_data_task_name(cfg.data_config, stage='prep')
    task, logger = init_task(project_name, task_dataset_name)
    if create_draft is True:
        task.execute_remotely()

    cfg = connect_cfg(task, cfg)
    data_cfg = cfg.data_config

    logger.report_text(f'Starting pre-processing of `{task_dataset_name}`.')

    data_manager = PreprocessDataManager(
        project_name=project_name,
        task_dataset_name=task_dataset_name,
        data_cfg=data_cfg,
        logger=logger,
    )
    raw_csv_path = get_raw_ds_local_path(data_manager.raw_dataset)
    processed_dir = preprocess_data(project_name, data_cfg, raw_csv_path, cfg.seed)
    logger.report_text('Pre-processing finished! Uploading data to ClearML...')
    ds = data_manager.upload_processed_ds(processed_dir)
    return ds


def get_prep_data(cfg: MLPExperimentConfig) -> Path:
    raw_dataset_name = get_data_task_name(cfg.data_config, stage='prep')
    return Path(
        Dataset.get(
            dataset_project=cfg.project_name,
            dataset_name=raw_dataset_name,
            alias='preprocessed_dataset',
        ).get_local_copy(),
    )


if __name__ == '__main__':
    cfg = get_experiment_cfg()
    clearml_preprocess(cfg, True)
