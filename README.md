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

______________________________________________________________________

# Main workflow

## Train MLP on [Heart Disease diagnosis dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

```bash
invoke train_mlp
```

Without arguments this command will use experiment config `configs/heart_mlp_config.yaml`.

The command above will execute the whole pipeline:

1. Download data
1. Preprocess data
1. Train MLP

**Preprocessing steps:**

1. Filter rows with non-positive values in selected columns that must be positive.
1. Tran/val/test split
1. Apply standardization to numeric variables
1. Apply one-hot encoding to categorical variables

### Specify path to experiment config

Experiment config must strictly follow `src/config.py`'s `Pydantic` model.

```bash
invoke train_mlp --cfg=configs/<your_cfg_name>.yaml
```

### Check training logs

All training logs are saved in `src/lightning_logs` directory and currently can be accessed using [TensorBoard](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.). Although experiment tracking in [ClearML](https://clear.ml/docs/latest/docs/) will be configured soon.

______________________________________________________________________

## More control. Development and debugging utils

### Jupyter Notebooks

#### Run Jupyter Lab

Configured to enable using CTRL+B to jump to code definitions

```bash
invoke jupyter_start
```

#### Briefly review raw (not pre-processed) dataset in the notebook

```bash
invoke jupyter_heart_review
```

### Run data processing and training steps separately

`--cfg` argument to all commands below is optional, MLP Heart Dataset config is used by default.

#### 1. Download Heart Disease dataset

```bash
   invoke download_heart --cfg=<path_to_your_cfg>
```

#### 2. Preprocess dataset

In fact, this command is dataset agnostic.

```bash
   invoke preprocess --cfg=<path_to_your_cfg>
```

#### 3. Download + preprocess dataset as during MLP training

You can use this command to test Lightning DataModule's `prepare_data()` method that runs downloading + pre-processing as during training.

```bash
   invoke get_data_mlp --cfg=<path_to_your_cfg>
```

______________________________________________________________________

## Development notes

### Invoke - CLI for scripts

Invoke is used here for task/command management. It allows to run any scripts seamlessly via CLI, both main functionalities (like data prep or training) and utilities.  All CLI commands (Invoke `tasks`) are defined in `tasks.py`, so you can find all existing commands here.

To make it easier to run main functionalities and utils, make sure to add a new `task` to `tasks.py` whenever you develop a new script or util.

______________________________________________________________________

## Acknowledgments

`Lightning` code in this project was inspired by [Egor's](https://github.com/EgorOs) awesome [ml_refactoring_lecture repo](https://github.com/EgorOs/ml_refactoring_lecture/tree/main).
