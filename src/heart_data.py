from pathlib import Path
from zipfile import ZipFile

from invoke import Context

from src.config import DataConfig


def download(ctx: Context, cfg: DataConfig) -> None:
    data_dir = Path(cfg.raw_csv_path).parent

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
