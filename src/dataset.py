from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import PROJECT_ROOT


class TabularDataset(Dataset):
    def __init__(self, path: Path, split: str, target_col: str):
        self.data = TabularSplit.from_folder(path, split, target_col)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


@dataclass(frozen=True)
class TabularSplit:
    _features: pd.DataFrame
    _target: pd.Series
    split: str

    @property
    def features(self) -> pd.DataFrame:
        return self._features.copy()

    @property
    def target(self) -> pd.Series:
        return self._target.copy()

    def __post_init__(self) -> None:
        if self.split not in ('train', 'val', 'test'):
            raise ValueError(f'`stage` must be either `train`, `val` or `test`, got `{self.split}`')

        features_len, target_len = len(self._features), len(self._target)
        if features_len != target_len:
            raise ValueError(
                'Length of `features` and `target` must be equal, got: length of features = '
                f'{features_len}, length of target = {target_len}',
            )

    def __len__(self) -> int:
        return len(self._target)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self._features.iloc[idx], dtype=torch.float16),
            torch.tensor(self._target.iloc[idx], dtype=torch.uint8),
        )

    def to_csv(self, export_dir: Union[str, Path]) -> None:
        export_path = PROJECT_ROOT / export_dir / self.split
        export_path.mkdir(parents=True, exist_ok=True)
        self._features.to_csv(export_path / 'features.csv', index=False)
        self._target.to_csv(export_path / 'target.csv', index=False)

    @classmethod
    def from_folder(cls, processed_path: Path, split: str, target_col: str) -> 'TabularSplit':
        subset_path = processed_path / split
        features = pd.read_csv(subset_path / 'features.csv')
        target = pd.read_csv(subset_path / 'target.csv')[target_col]
        return TabularSplit(features, target, split)
