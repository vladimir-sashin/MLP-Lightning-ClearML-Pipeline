from dataclasses import dataclass
from pathlib import Path
from typing import Union

from src.data.data_model import TabularSplit


@dataclass(frozen=True)
class TabularSplitsCollection:
    train: TabularSplit
    val: TabularSplit
    test: TabularSplit

    def to_csv(self, export_dir: Union[str, Path]) -> None:
        self.train.to_csv(export_dir)
        self.val.to_csv(export_dir)
        self.test.to_csv(export_dir)
