import os
from pathlib import Path
from typing import Type, TypeVar, Union

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator

from src.constants import DEFAULT_HEART_DATA_CFG_PATH

T = TypeVar('T', bound='_BaseValidatedConfig')


class _BaseValidatedConfig(BaseModel):  # type: ignore
    model_config = ConfigDict(extra='forbid')  # Disallow unexpected arguments.

    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)


class HeartDataConfig(_BaseValidatedConfig):
    dataset_name: str
    input_path: str
    output_dir: str
    seed: int
    test_ratio: float

    @model_validator(mode='after')
    def test_ratio_leq_one(self) -> 'HeartDataConfig':
        if self.test_ratio > 0.5:
            raise ValueError(f'Test set ratio should be less than or equal to 0.5, got {self.test_ratio}')
        return self


def get_heart_data_config() -> HeartDataConfig:
    cfg_path = os.getenv('HEART_DATA_CFG_PATH', DEFAULT_HEART_DATA_CFG_PATH)
    return HeartDataConfig.from_yaml(cfg_path)
