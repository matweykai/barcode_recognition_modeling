CALL_CMD=PYTHONPATH=. python
ACTIVATE_VENV=source .venv/bin/activate
CONFIG_PATH=configs/base_config.yaml

SHELL := /bin/bash
.ONESHELL:

setup:
	python3 -m venv .venv
	$(ACTIVATE_VENV)
	pip install -r requirements.txt
	dvc install
	dvc pull
	clearml-init

train:
	$(ACTIVATE_VENV)
	$(CALL_CMD) src/train.py $(CONFIG_PATH)

MODEL_CKPT_PATH=weights/eff_net_b5_loss=0.196.ckpt
export_onnx:
	$(ACTIVATE_VENV)
	$(CALL_CMD) src/export.py $(MODEL_CKPT_PATH)
