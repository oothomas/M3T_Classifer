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

## Running

```bash
python scripts/pretrain.py configs/pretrain.yaml
python scripts/finetune.py configs/finetune.yaml
python scripts/saliency.py configs/saliency.yaml
```

Adjust the YAML files with the appropriate data lists and checkpoint paths
before running.
