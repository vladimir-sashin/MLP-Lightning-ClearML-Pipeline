from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve

import lightning

from src.config import DataConfig
from src.data.preprocessing.path_helpers import _get_dataset_dir, _get_processed_dir_path
from src.data.preprocessing.steps import (
    filter_positive_cols,
    fit_col_transformer,
    read_data,
    split_data,
    transform_cols,
)

RAW_CSV_FILENAME = 'raw.csv'


def get_raw_csv_path(project_name: str, data_cfg: DataConfig) -> Path:
    return _get_dataset_dir(project_name, data_cfg.orig_dataset_name) / RAW_CSV_FILENAME


def preprocess_data(
    project_name: str,
    data_cfg: DataConfig,
    raw_csv_path: Union[Path, str],
    seed: Optional[int] = None,
) -> Path:
    """Preprocess a CSV dataset using ProcessingConfig

    Args:
        cfg: ProcessingConfig that describes all paths and transformations
        project_name: Name of the project to which the dataset belongs
        raw_csv_path: Path to the raw data .csv file
        seed: If passed, sets random seed. If not passed, make sure to set seed before to get reproducible results
    """
    prep_cfg = data_cfg.processing_config
    print(f'Initiating dataset preprocessing using the following config: {prep_cfg}')
    if seed is not None:
        lightning.seed_everything(seed)

    features, target = read_data(Path(raw_csv_path), prep_cfg.target_column)
    features, target = filter_positive_cols(features, target, prep_cfg.positive_columns)
    splits = split_data(features, target, prep_cfg.split_ratios)

    transformer = fit_col_transformer(
        splits.train.features,
        prep_cfg.categorical_columns,
        prep_cfg.apply_standardization,
    )
    splits = transform_cols(transformer, splits)

    processed_dir = _get_processed_dir_path(project_name, data_cfg.orig_dataset_name)
    splits.to_csv(processed_dir)

    print(f'Dataset is preprocessed and saved to {processed_dir}')
    return processed_dir


def download_csv(project_name: str, data_cfg: DataConfig, skip_if_exists: bool = False) -> Path:
    """Download CSV dataset from direct URL

    Args:
        project_name: Used to determine a path where file will be downloaded
        data_cfg: ProcessingConfig instance. Used fields:
            cfg.raw_csv_url: Direct URL to the single .csv file that will be downloaded
            cfg.orig_dataset_name: Used to determine a path where file will be downloaded
        skip_if_exists: If True, check if file is already downloaded to the determined path

    Returns:
        Path where file is downloaded: `dataset_dir/raw.csv`
        (check `_get_dataset_dir()` func to see how `dataset_dir` path is generated)
    """
    raw_csv_path = get_raw_csv_path(project_name, data_cfg)
    if skip_if_exists is True:
        if raw_csv_path.is_file():
            print(f'Raw `{data_cfg.orig_dataset_name}` dataset already exists and won\'t be downloaded.')
            return raw_csv_path
    print(f'Downloading raw `{data_cfg.orig_dataset_name}` dataset...')
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(data_cfg.raw_csv_url, raw_csv_path)
    print(f'Raw `{data_cfg.orig_dataset_name}` dataset is downloaded to the `{raw_csv_path}` path...')
    return raw_csv_path
