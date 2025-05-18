import argparse, yaml
from training.pretrain import run_pretraining

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run SSL pretraining')
    ap.add_argument('config', help='Path to YAML config')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_pretraining(cfg)
