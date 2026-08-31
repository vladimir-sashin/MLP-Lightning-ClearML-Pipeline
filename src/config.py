import os
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Tuple, Type, TypeVar, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.constants import MLP_CFG_PATH

T = TypeVar('T', bound='_BaseValidatedConfig')


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)  # Disallow unexpected arguments.


class _ConfigYamlMixin(BaseModel):
    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, 'w') as out_file:
            yaml.safe_dump(self.model_dump(), out_file, default_flow_style=False, sort_keys=False)


class SplitRatios(NamedTuple):
    train: float
    val: float
    test: float


class DataLoaderConfig(_BaseValidatedConfig):
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True


class ProcessingConfig(_BaseValidatedConfig):
    split_ratios: SplitRatios = SplitRatios(0.7, 0.15, 0.15)
    target_column: str = 'HeartDisease'
    categorical_columns: Optional[Tuple[str, ...]] = (
        None  # all other columns except target will be treated as numerical
    )
    positive_columns: Optional[Tuple[str, ...]] = None  # values <= 0 will be filtered out
    apply_standardization: bool = True

    @model_validator(mode='after')
    def splits_add_up_to_one(self) -> 'ProcessingConfig':
        epsilon = 1e-5
        total = sum(self.split_ratios)
        if abs(total - 1) > epsilon:
            raise ValueError(f'Splits should add up to 1, got {total}.')
        return self


class DataConfig(_BaseValidatedConfig):
    orig_dataset_name: str = 'heart_disease_dataset_mlp'
    dataset_description: str = 'Heart Disease dataset'
    raw_csv_url: str = 'https://drive.google.com/u/0/uc?id=1zm4NnDrVhIKH9-fatlD5idECalFyWK0y&export=download'
    processing_config: ProcessingConfig = Field(default=ProcessingConfig())


class MLPTrainerConfig(_BaseValidatedConfig):
    min_epochs: int = 7  # prevents early stopping
    max_epochs: int = 20

    # perform a validation loop every N training epochs
    check_val_every_n_epoch: int = 3

    log_every_n_steps: int = 50

    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = None

    # set True to ensure deterministic results
    # makes training slower but gives more reproducibility than just setting seeds
    deterministic: bool = False

    fast_dev_run: bool = False
    default_root_dir: Optional[Path] = None

    detect_anomaly: bool = False


class MLPModelConfig(_BaseValidatedConfig):
    linear_1_dim: int = 500
    linear_2_dim: int = 500


class MLPHyperparametersConfig(_BaseValidatedConfig):
    lr: float = 2e-3


class RunModeEnum(str, Enum):
    pipeline = 'pipeline'
    local = 'local'


class MLPExperimentConfig(_BaseValidatedConfig, _ConfigYamlMixin):
    project_name: str = 'mlp_classification'
    seed: int = 42
    experiment_name: str = 'training'
    track_in_clearml: bool = False
    run_mode: RunModeEnum = RunModeEnum.pipeline
    data_config: DataConfig = Field(default=DataConfig())
    dataloader_config: DataLoaderConfig = Field(default=DataLoaderConfig())
    trainer_config: MLPTrainerConfig = Field(default=MLPTrainerConfig())
    mlp_model_config: MLPModelConfig = Field(default=MLPModelConfig())
    hyperparameters_config: MLPHyperparametersConfig = Field(default=MLPHyperparametersConfig())


def get_experiment_cfg(cfg_path: Optional[Union[str, Path]] = None) -> MLPExperimentConfig:
    cfg_path = cfg_path or os.getenv('TRAIN_MLP_CFG_PATH')
    if not cfg_path:
        return MLPExperimentConfig.from_yaml(MLP_CFG_PATH)
    return MLPExperimentConfig.from_yaml(cfg_path)
