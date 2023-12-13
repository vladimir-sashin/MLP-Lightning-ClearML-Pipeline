from clearml import PipelineController

from src.clearml_pipeline.init_data.core import get_init_cfg_dump
from src.clearml_pipeline.preprocess.core import get_prep_cfg_dump
from src.clearml_pipeline.utils import get_data_task_name
from src.config import MLPExperimentConfig, get_experiment_cfg


def _setup_pipeline(cfg: MLPExperimentConfig) -> PipelineController:
    pipe = PipelineController(
        project=cfg.project_name,
        name=f'{cfg.project_name} pipeline',
        add_pipeline_tags=False,
        target_project=cfg.project_name,
    )
    pipe.set_default_execution_queue('default')
    return pipe


def _connect_cfg(pipe: PipelineController, cfg: MLPExperimentConfig) -> MLPExperimentConfig:
    cfg_dump = cfg.model_dump()
    pipe.connect_configuration(cfg_dump)
    cfg = MLPExperimentConfig.model_validate(cfg_dump)
    return cfg


def run_pipeline(cfg: MLPExperimentConfig) -> None:
    pipe = _setup_pipeline(cfg)
    cfg = _connect_cfg(pipe, cfg)

    init_task_name = get_data_task_name(cfg.data_config, stage='init')
    prep_task_name = get_data_task_name(cfg.data_config, stage='prep')

    pipe.add_step(
        name=init_task_name,
        base_task_project=cfg.project_name,
        base_task_name=init_task_name,
        configuration_overrides={'General': get_init_cfg_dump(cfg.data_config)},
        recursively_parse_parameters=True,
        cache_executed_step=True,
    )

    pipe.add_step(
        name=prep_task_name,
        base_task_project=cfg.project_name,
        base_task_name=prep_task_name,
        parents=[init_task_name],
        configuration_overrides={'General': get_prep_cfg_dump(cfg)},
        recursively_parse_parameters=True,
        cache_executed_step=True,
    )

    pipe.add_step(
        name=cfg.experiment_name,
        base_task_project=cfg.project_name,
        base_task_name=cfg.experiment_name,
        parents=[prep_task_name],
        monitor_models=['*'],
        recursively_parse_parameters=True,
        configuration_overrides={'General': cfg.model_dump()},
    )

    pipe.start_locally(run_pipeline_steps_locally=True)


if __name__ == '__main__':
    cfg = get_experiment_cfg()
    run_pipeline(cfg)
