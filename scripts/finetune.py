"""Entry point script for classifier fine-tuning."""

import argparse
import yaml
from training.finetune import (
    run_finetuning_edema,
    run_finetuning_exencephaly,
    run_finetuning_gli2,
)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run classifier finetuning')
    ap.add_argument('config', help='Path to YAML config')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = cfg.get('task', 'edema')
    if task == 'edema':
        run_finetuning_edema(cfg)
    elif task == 'exencephaly':
        run_finetuning_exencephaly(cfg)
    elif task == 'gli2':
        run_finetuning_gli2(cfg)
    else:
        raise ValueError(f"Unknown finetuning task: {task}")
