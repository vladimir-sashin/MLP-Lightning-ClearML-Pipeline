import os
from sys import platform
from typing import Optional

from invoke import Context, task

from src.config import get_heart_data_config
from src.constants import HEART_NOTEBOOKS, PROJECT_ROOT
from src.preprocess_heart import download_data, preprocess


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
def heart_download(ctx: Context) -> None:
    cfg = get_heart_data_config()
    download_data(cfg, ctx)


@task
def heart_preprocess(ctx: Context) -> None:
    cfg = get_heart_data_config()
    preprocess(cfg)


@task
def heart_data_pipe(ctx: Context) -> None:
    heart_download(ctx)
    heart_preprocess(ctx)


@task
def jupyter_heart_review(ctx: Context) -> None:
    notebook_path = HEART_NOTEBOOKS / 'heart_data_review.ipynb'
    jupyter_start(ctx, notebook=notebook_path)
