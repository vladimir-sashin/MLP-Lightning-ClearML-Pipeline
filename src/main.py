from src.clearml_pipeline.pipeline import run_pipeline
from src.clearml_pipeline.train_task import clearml_train_mlp
from src.config import MLPExperimentConfig, RunModeEnum, get_experiment_cfg
from src.train.train import train_mlp


def run_training(cfg: MLPExperimentConfig) -> None:
    if cfg.run_mode == RunModeEnum.pipeline:
        return run_pipeline(cfg)
    elif cfg.run_mode == RunModeEnum.local:
        if cfg.track_in_clearml is True:
            return clearml_train_mlp(cfg)
        return train_mlp(cfg)


if __name__ == '__main__':
    cfg = get_experiment_cfg()
    run_training(cfg)
