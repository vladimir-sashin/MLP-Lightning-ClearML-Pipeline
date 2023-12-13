from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.data.data_model import TabularSplit


class TabularDataset(Dataset):
    def __init__(self, path: Path, split: str, target_col: str):
        self.data = TabularSplit.from_folder(path, split, target_col)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

    @property
    def num_classes(self) -> int:
        return len(self.data.target.unique())

    @property
    def num_features(self) -> int:
        return len(self.data.features.columns)
