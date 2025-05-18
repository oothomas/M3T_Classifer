import argparse, yaml
from saliency.generate_maps import generate_maps

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Generate saliency maps')
    ap.add_argument('config', help='Path to YAML config')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    generate_maps(cfg)
