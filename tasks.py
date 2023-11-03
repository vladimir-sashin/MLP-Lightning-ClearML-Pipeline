from sys import platform

from invoke import task


@task
def jupyter_start(ctx):
    if platform == 'win32':
        print('Skipped lsp symlink setup for Windows')
    else:
        ctx.run('ln -s / .lsp_symlink || true')
    ctx.run('jupyter nbextension enable --py --sys-prefix widgetsnbextension')
    ctx.run('jupyter lab --ContentsManager.allow_hidden=True')
