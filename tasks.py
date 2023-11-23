import os
from pathlib import Path
from sys import platform
from typing import Optional, Union

from invoke import Context, task

from src import heart_data, train
from src.config import get_experiment_cfg
from src.constants import HEART_NOTEBOOKS, MLP_CFG_PATH, PROJECT_ROOT
from src.datamodule import dm_prepare_data
from src.preprocessing import preprocess_data


@task
def jupyter_start(ctx: Context, notebook: Optional[str] = None) -> None:
    # These lines ensure that CTRL+B can be used to jump to definitions in
    # code of installed modules on Linux/OSX.
    # Explained here:
    # https://github.com/jupyter-lsp/jupyterlab-lsp/blob/39ee7d93f98d22e866bf65a80f1050d67d7cb504/README.md?plain=1#L175
    if platform == 'win32':
        print('Skipped lsp symlink setup for Windows')
    else:
        ctx.run('ln -s / .lsp_symlink || true')

    jupyter_args = '--ContentsManager.allow_hidden=True'

    os.environ['PYTHONPATH'] = str(PROJECT_ROOT)
    ctx.run('jupyter nbextension enable --py --sys-prefix widgetsnbextension')
    if notebook:
        ctx.run(f'jupyter lab {notebook} {jupyter_args}')
    else:
        ctx.run(f'jupyter lab {jupyter_args}')


@task
def jupyter_heart_review(ctx: Context) -> None:
    notebook_path = HEART_NOTEBOOKS / 'heart_data_review.ipynb'
    jupyter_start(ctx, notebook=notebook_path)


@task
def download_heart(ctx: Context, cfg: Optional[Union[str, Path]] = None) -> None:
    data_cfg = get_experiment_cfg(cfg).data_config
    heart_data.download(ctx, data_cfg)


@task
def preprocess(ctx: Context, cfg: Optional[Union[str, Path]] = None) -> None:
    exp_cfg = get_experiment_cfg(cfg)
    preprocess_data(exp_cfg.data_config, seed=exp_cfg.seed)


@task
def get_data_mlp(ctx: Context, cfg: Union[str, Path] = MLP_CFG_PATH) -> None:
    exp_cfg = get_experiment_cfg(cfg)
    dm_prepare_data(exp_cfg.data_config, exp_cfg.seed)


@task
def train_mlp(ctx: Context, cfg: Optional[Union[str, Path]] = MLP_CFG_PATH) -> None:
    exp_cfg = get_experiment_cfg(cfg)
    train.train_mlp(exp_cfg)
