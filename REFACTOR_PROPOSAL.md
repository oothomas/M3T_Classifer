# Proposed Refactor

This document summarizes the current scripts and proposes a modular layout suitable for three stages: pretraining, finetuning, and saliency map generation.

## Current Structure

- `pretraining_script.py` – large script implementing BYOL-style self–supervised learning with extra heads for reconstruction, rotation, and jigsaw tasks.
- `finetuning_script.py` – loads an SSL checkpoint and trains a binary edema classifier using the same encoder architecture.
- `saliency_script.py` – loads the fine-tuned model and produces saliency maps via Captum.

Each script defines the model components, data transforms, and training/evaluation logic locally, leading to duplication and difficult maintenance.

## Goals

1. **Share common components** such as the encoder, augmentation utilities, and training loops.
2. **Separate configuration** from code so that different experiments reuse the same modules.
3. **Provide lightweight entry scripts** for each stage that import the shared modules.

## Suggested File Layout

```
MT3_Classifier/
├── models/
│   ├── encoder.py            # CNN3D + multi-plane projector + embedding + transformer
│   ├── ssl_heads.py          # BYOL wrapper, reconstruction decoder, rotation and jigsaw heads
│   └── classifier.py         # Edema classification head
│
├── data/
│   ├── transforms.py         # build_transforms, augment_batch, create_mask
│   └── dataset.py            # Dataset wrappers for NRRD files
│
├── training/
│   ├── pretrain.py           # function `run_pretraining(config)`
│   ├── finetune.py           # function `run_finetuning(config)`
│   └── utils.py              # lr_schedule and helper functions
│
├── saliency/
│   └── generate_maps.py      # reusable saliency functions using Captum
│
├── configs/
│   ├── pretrain.yaml         # hyper-parameters for SSL
│   ├── finetune.yaml         # hyper-parameters for classifier training
│   └── saliency.yaml         # paths and options for saliency generation
│
├── scripts/
│   ├── pretrain.py           # parses CLI args and calls `training.pretrain`
│   ├── finetune.py           # CLI entry point for fine-tuning
│   └── saliency.py           # CLI entry for saliency generation
└── ...
```

## Benefits

- **Reusability** – the same encoder definition is imported by all stages.
- **Smaller entry scripts** – they mainly load a config and call the appropriate training or evaluation function.
- **Simpler experimentation** – changing hyper-parameters only requires editing a YAML file.
- **Extensibility** – new tasks or losses can be added by extending modules in `models/` or `training/`.

## Next Steps

1. Extract the model definitions from the existing scripts into `models/encoder.py` and `models/ssl_heads.py`.
2. Move data augmentation utilities into `data/transforms.py`.
3. Implement `training/pretrain.py` and `training/finetune.py` that accept a config object and encapsulate the loops currently found in the scripts.
4. Provide CLI wrappers in `scripts/` that load YAML configs and invoke the training functions.
5. Update the README with instructions on running each stage with the new layout.

This refactor would streamline experimentation by reducing code duplication and clarifying the responsibilities of each module.
