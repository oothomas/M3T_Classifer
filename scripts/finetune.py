"""Entry point script for classifier fine-tuning."""

import argparse
import yaml
from training.finetune import run_finetuning

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run classifier finetuning')
    ap.add_argument('config', help='Path to YAML config')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_finetuning(cfg)
