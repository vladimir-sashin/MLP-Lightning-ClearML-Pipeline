from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Type, TypeVar, Union

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator

from src.constants import DEFAULT_DATA_CFG_PATH

T = TypeVar('T', bound='_BaseValidatedConfig')


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')  # Disallow unexpected arguments.

    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)


class SplitRatios(NamedTuple):
    train: float
    val: float
    test: float


class DataConfig(_BaseValidatedConfig):
    dataset_name: str
    raw_csv_path: str
    processed_path: str
    seed: int
    split_ratios: SplitRatios = SplitRatios(0.7, 0.15, 0.15)
    target_column: str
    categorical_columns: Optional[
        Tuple[str, ...]
    ] = None  # all other columns except target will be treated as numerical
    positive_columns: Optional[Tuple[str, ...]] = None  # values <= 0 will be filtered out
    apply_standardization: bool

    @model_validator(mode='after')
    def splits_add_up_to_one(self) -> 'DataConfig':
        epsilon = 1e-5
        total = sum(self.split_ratios)
        if abs(total - 1) > epsilon:
            raise ValueError(f'Splits should add up to 1, got {total}.')
        return self


def get_data_cfg(cfg_path: Optional[Union[str, Path]] = None) -> DataConfig:
    if not cfg_path:
        return DataConfig.from_yaml(DEFAULT_DATA_CFG_PATH)
    return DataConfig.from_yaml(cfg_path)
