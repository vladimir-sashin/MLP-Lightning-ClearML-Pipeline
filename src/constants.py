from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / 'configs'

DEFAULT_DATA_CFG_PATH = CONFIGS / 'mlp_heart_data_config.yaml'

MLP_HEART_DATA_CFG_PATH = CONFIGS / 'mlp_heart_data_config.yaml'

HEART_NOTEBOOKS = PROJECT_ROOT / 'notebooks' / 'heart_disease'
