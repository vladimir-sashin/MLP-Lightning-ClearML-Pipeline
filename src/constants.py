from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / 'configs'

MLP_CFG_PATH = CONFIGS / 'heart_mlp_config.yaml'

HEART_NOTEBOOKS = PROJECT_ROOT / 'notebooks' / 'heart_disease'
