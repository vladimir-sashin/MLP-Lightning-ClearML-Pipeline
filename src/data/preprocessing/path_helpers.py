from pathlib import Path

from src.constants import TMP_DATA_DIR


def _get_dataset_dir(project_name: str, orig_dataset_name: str) -> Path:
    return TMP_DATA_DIR / project_name / orig_dataset_name


def _get_processed_dir_path(project_name: str, orig_dataset_name: str) -> Path:
    dataset_dir = _get_dataset_dir(project_name, orig_dataset_name)
    return dataset_dir / 'processed'
