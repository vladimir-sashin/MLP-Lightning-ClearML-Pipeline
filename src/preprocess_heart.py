from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union
from zipfile import ZipFile

import lightning
import pandas as pd
from invoke import Context
from sklearn.model_selection import train_test_split

from src.config import HeartDataConfig
from src.constants import PROJECT_ROOT


@dataclass
class HeartSubset:
    features: pd.DataFrame
    target: pd.Series
    subset_type: str

    def __post_init__(self) -> None:
        if self.subset_type not in ('train', 'test'):
            raise ValueError(f'`subset_type` must be either `train` or `test`, got `{self.subset_type}`')

    def to_csv(self, export_dir: Union[str, Path]) -> None:
        export_dir = PROJECT_ROOT / Path(export_dir) / self.subset_type
        export_dir.mkdir(parents=True, exist_ok=True)
        self.features.to_csv(export_dir / 'features.csv', index=False)
        self.target.to_csv(export_dir / 'target.csv', index=False)


@dataclass
class HeartPreprocessed:
    train_set: HeartSubset
    test_set: HeartSubset

    def to_csv(self, export_dir: Union[str, Path]) -> None:
        self.train_set.to_csv(export_dir)
        self.test_set.to_csv(export_dir)


def download_data(cfg: HeartDataConfig, ctx: Context) -> None:
    data_dir = Path(cfg.input_path).parent

    kaggle_dataset_name = 'heart-failure-prediction'
    dataset_zip = f'{kaggle_dataset_name}.zip'
    kaggle_download_command = f'kaggle datasets download -d fedesoriano/{kaggle_dataset_name}'

    print('Downloading dataset from kaggle...')
    print(kaggle_download_command)
    ctx.run(kaggle_download_command)

    data_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f'Successfully downloaded and extracted data to {data_dir}')
    Path(dataset_zip).unlink()


def _read_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(PROJECT_ROOT / csv_path)
    data = data[(data['Cholesterol'] > 0) & (data['RestingBP'] > 0)]
    features = data.drop(['HeartDisease'], axis=1)
    target = data['HeartDisease']
    return features, target


def _split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_ratio: float,
    seed: int,
) -> HeartPreprocessed:
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_ratio, random_state=seed)
    return HeartPreprocessed(HeartSubset(x_train, y_train, 'train'), HeartSubset(x_test, y_test, 'test'))


def preprocess(cfg: HeartDataConfig) -> None:
    print(f'Initiating dataset preprocessing using the following config: {cfg}')
    lightning.seed_everything(cfg.seed)

    features, target = _read_data(cfg.input_path)
    preprocessed_data = _split_data(features, target, cfg.test_ratio, cfg.seed)
    preprocessed_data.to_csv(cfg.output_dir)

    print(f'Dataset is preprocessed and saved to {cfg.output_dir}')
