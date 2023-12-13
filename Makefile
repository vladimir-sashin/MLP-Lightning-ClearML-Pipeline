.PHONY: *

# If OS is Windows, path to Python 3.10 executable must be passed in `PYTHON_EXEC` var to `make setup_ws`
# Because poetry fails on `poetry env use pythonX.Y` command on Windows
# https://github.com/python-poetry/poetry/issues/2117
ifndef PYTHON_EXEC
override PYTHON_EXEC = python3.10
endif


# These lines ensure that CTRL+B can be used to jump to definitions in
# code of installed modules on Linux/OSX.
# Explained here: https://github.com/jupyter-lsp/jupyterlab-lsp/blob/39ee7d93f98d22e866bf65a80f1050d67d7cb504/README.md?plain=1#L175
ifeq ($(OS),Windows_NT)
	CREATE_LINK := @echo "Skipped lsp symlink setup for Windows"
else
	CREATE_LINK := ln -s / .lsp_symlink || true  # Create if does not exist.
endif


# ================== LOCAL WORKSPACE SETUP ==================
setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install --with notebooks
	poetry run pre-commit install
	@echo "Virtual environment has been created."
	@echo "Path to virtual environment:"
	poetry env info -p


# =================== CLEARML INIT TASKS ====================
# IMPORTANT! If you want to run the training in the pipeline mode
# ClearML pipeline orchestrates and executes tasks, and to be able to run the pipeline, you firstly need to init tasks
# that are operated by the pipeline. To do so, run a command corresponding to the ClearML task whenever you change:
# 	- source code used by this ClearML task
#	- `project_name` or `data_config.orig_dataset_name` values in the config
#	- `experiment_name` value in the config (only for the training task)

clearml_init_task_data:
	poetry run python src/clearml_pipeline/init_data/task.py

clearml_init_prep:
	poetry run python src/clearml_pipeline/preprocess/task.py

clearml_init_train:
	poetry run python src/clearml_pipeline/train_task.py

clearml_init_tasks:
	# Init all tasks at once
	# Note: If you run this command for your project for the 1st time, ClearML will print 2 errors that the project
	# already exists, this is expected as targets are executed concurrently to speed up the setup
	$(MAKE) clearml_init_task_data & \
	$(MAKE) clearml_init_prep & \
	$(MAKE) clearml_init_train


# ========================= TRAINING ========================
# Training modes can be configured in the config file
# `run_mode`: `pipeline` or `local`
# 	- `pipeline` runs ClearML pipeline (tasks must already exist in ClearML). Everything is tracked, all artifacts are
# 		uploaded in Clearml. IMPORTANT: check `CLEARML INIT TASKS` section before running.
#	- `local` also runs the whole pipeline, but not as a ClearML pipeline. Additionally, if `track_in_clearml` is`true`,
# 		experiment is tracked in ClearML, model artifacts are also uploaded, but data artifacts are not.
#		In case `track_in_clearml` is`false`, nothing is tracked or uploaded to ClearML.
run_training:
	# Set `TRAIN_MLP_CFG_PATH` env variable to provide an absolute path to the config.
	# If not provided, default config is used (check `src/config.py`)
	poetry run python src/main.py


# ========================= JUPYTER =========================
jupyterlab_start:
	$(CREATE_LINK)
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	jupyter lab --ContentsManager.allow_hidden=True
