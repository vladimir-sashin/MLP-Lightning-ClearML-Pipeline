# Topological XAI research

## Getting started

1. Follow [instructions](https://github.com/python-poetry/install.python-poetry.org) to install Poetry.
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
1. Activate poetry virtual environment
   ```bash
   poetry shell
   ```
1. Follow [instructions](https://www.kaggle.com/docs/api#authentication) to setup kaggle credentials to download initial dataset.
1. \[Optional\] Follow [instructions](https://docs.pyinvoke.org/en/1.0/invoke.html#shell-tab-completion) to configure Invoke commands tab completion.

# Main workflow

## Simple

### MLP training

1. **Heart Disease dataset preprocessing pipeline for MLP training**. Download Heart Disease dataset and preprocess it using `configs/mlp_heart_data_config.yaml` config:

   1. Filter rows with non-positive values in selected columns that must be positive.
   1. Tran/val/test split
   1. Apply standardization to numeric variables
   1. Apply one-hot encoding to categorical variables

   ```bash
    invoke data_pipeline_heart_mlp
   ```

1. Run notebook to briefly review the data:

   ```bash
   invoke jupyter_heart_review
   ```

## Full control

### Manually specify config path

1. Create or select a data config (must be valid, check `src/config.py`)
1. Set path to data config in the terminal, for example:
   - `bash`:
     ```bash
     cfg=configs/mlp_heart_data_config.yaml
     ```
   - `Windows PowerShell`:
     ```bash
     $cfg=configs\mlp_heart_data_config.yaml
     ```
1. Run Heart Disease dataset preprocessing pipeline:
   ```bash
   invoke data_pipeline_heart --data-cfg=$cfg
   ```

### Run pipeline stages separately

1. Download Heart Disease dataset:

   ```bash
   invoke download_heart --data-cfg=$cfg
   ```

1. Preprocess data:

   ```bash
   invoke preprocess --data-cfg=$cfg
   ```

## Development

### Invoke - CLI for scripts

Invoke is used for task/command management. It allows to run any scripts seamlessly via CLI, both main functionalities (like data prep or training) and utilities.  All CLI commands (Invoke `tasks`) are defined in `tasks.py`, so you can find all existing commands here.

To make it easier to run main functionalities and utils, make sure to add a new `task` to `tasks.py` whenever you develop a new script or util.

### Jupyter Lab

Run Jupyter Lab (configured to enable using CTRL+B to jump to code definitions):

```bash
invoke jupyter_start
```
