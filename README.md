# Topological XAI research

## Initial repo setup

1. Follow [instructions](https://github.com/python-poetry/install.python-poetry.org) to install Poetry
1. Check that poetry was installed successfully:
   ```bash
   poetry --version
   ```
1. Choose Python 3.10 interpreter for poetry:
   - Unix:
     ```bash
     poetry env use python3.10
     ```
   - Windows:
     ```bash
     # Because poetry fails on `poetry env use pythonX.Y` command on Windows
     # https://github.com/python-poetry/poetry/issues/2117
     poetry env use <your_python_3.10_executable>
     ```
1. Install dependencies:
   ```bash
   poetry install --with notebooks
   ```
1. Path to the poetry virtual env:
   ```bash
   poetry env info -p
   ```
1. Install pre-commit:
   ```bash
   poetry run pre-commit install
   ```
