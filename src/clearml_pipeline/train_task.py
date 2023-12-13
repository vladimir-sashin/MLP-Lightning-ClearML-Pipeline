from clearml import Task

from src.config import MLPExperimentConfig, get_experiment_cfg
from src.train.train import train_mlp


def clearml_train_mlp(cfg: MLPExperimentConfig, create_draft: bool = False) -> None:
    if cfg.track_in_clearml:
        Task.force_requirements_env_freeze()
        task = Task.init(
            project_name=cfg.project_name,
            task_name=cfg.experiment_name,
            output_uri=True,  # If `output_uri=True` uses default ClearML output URI
        )
        if create_draft is True:
            task.execute_remotely()
        cfg_dump = cfg.model_dump()
        task.connect_configuration(configuration=cfg_dump)
        cfg = MLPExperimentConfig.model_validate(cfg_dump)

    train_mlp(cfg)


if __name__ == '__main__':
    cfg = get_experiment_cfg()
    clearml_train_mlp(cfg, True)
