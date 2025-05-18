# MT3_Classifier

This repository contains a modular implementation of the 3-plane CNN + Transformer
pipeline for embryonic CT analysis. The code is organised into reusable modules
for pretraining, finetuning and saliency map generation.

## Directory Overview

- `models/` – network architectures shared across tasks
- `data/` – data transforms and dataset wrappers
- `training/` – routines for pretraining and finetuning
- `saliency/` – utilities for generating saliency maps with Captum
- `scripts/` – entry points that load a YAML config and launch a job
- `configs/` – example configuration files

## Usage

The project is configured through YAML files located in `configs/`. Each script
expects a path to one of these files.

### Self-supervised pretraining

```bash
python scripts/pretrain.py configs/pretrain.yaml
```

Edit `configs/pretrain.yaml` so that `data_list` points to a list of training
volumes. The script will save checkpoints under `ssl_runs/` and log metrics via
Weights & Biases.

### Finetuning on edema labels

```bash
python scripts/finetune.py configs/finetune.yaml
```

`configs/finetune.yaml` requires paths to the `train_list` and `val_list` JSON
files. Optionally provide a pretrained checkpoint via `ssl_ckpt` to initialise
the encoder weights.

### Saliency map generation

```bash
python scripts/saliency.py configs/saliency.yaml
```

Specify the classifier checkpoint via `ckpt` and the dataset list via
`data_list`. Saliency maps will be written as NumPy arrays to the directory
given in `out_dir`.
