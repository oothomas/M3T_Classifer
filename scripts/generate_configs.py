import argparse
import os
import glob
import random
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ToTensord
)
from monai.data import Dataset


def str2bool(val):
    if pd.isna(val):
        return None
    if isinstance(val, bool):
        return val
    val = str(val).strip().lower()
    if val in {'true', '1', 'yes', 'y', 't'}:
        return True
    if val in {'false', '0', 'no', 'n', 'f'}:
        return False
    return None


def load_base_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_volume(scan_id, data_dir):
    pattern = os.path.join(data_dir, f"{scan_id}*.nrrd")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def build_datalist(csv_path, data_dir, task):
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        scan_id = str(row['ScanID']).strip()
        wt = str2bool(row['Wildtype'])
        phen = str2bool(row['Phenotype'])
        label = None
        if phen is True:
            label = 1
        elif phen is False and wt is True:
            label = 0
        if label is None:
            continue
        path = find_volume(scan_id, data_dir)
        if path is None:
            continue
        items.append({'image': path, f'label_{task}': label})
    return items


def split_data(items, val_split, seed=0):
    """Split items into train/val sets while keeping class balance."""

    if not items:
        return [], []

    label_key = next((k for k in items[0] if k.startswith("label_")), None)
    rng = random.Random(seed)

    if label_key is None:
        rng.shuffle(items)
        val_count = int(len(items) * val_split)
        return items[val_count:], items[:val_count]

    grouped = {}
    for item in items:
        grouped.setdefault(item[label_key], []).append(item)

    train_list = []
    val_list = []
    for _, subset in grouped.items():
        rng.shuffle(subset)
        val_count = int(round(len(subset) * val_split))
        val_list.extend(subset[:val_count])
        train_list.extend(subset[val_count:])

    rng.shuffle(train_list)
    rng.shuffle(val_list)
    return train_list, val_list


def compute_stats(items):
    """Compute global mean and std of the given dataset list."""

    tf = Compose([
        LoadImaged("image", image_only=False, reader="ITKReader"),
        EnsureChannelFirstd("image", strict_check=False),
        Orientationd("image", axcodes="SAR"),
        ToTensord("image")
    ])
    ds = Dataset(data=items, transform=tf)

    running_sum = 0.0
    running_sq_sum = 0.0
    voxel_count = 0

    for i in tqdm(range(len(ds)), desc="Computing mean/std"):
        vol = ds[i]["image"].float()
        v = vol.view(-1)
        running_sum += float(v.sum())
        running_sq_sum += float((v ** 2).sum())
        voxel_count += v.numel()

    mean = running_sum / voxel_count
    var = (running_sq_sum / voxel_count) - mean ** 2
    std = float(np.sqrt(var))
    return float(mean), std


def main():
    ap = argparse.ArgumentParser(description='Generate finetuning config')
    ap.add_argument('--task', required=True, choices=['edema', 'exencephaly', 'gli2'],
                    help='Target classification task')
    ap.add_argument('--csv', required=True, help='CSV file with labels')
    ap.add_argument('--data-dir', required=True, help='Directory with NRRD volumes')
    ap.add_argument('--base-config', default=os.path.join('configs', 'finetune.yaml'),
                    help='Base YAML config')
    ap.add_argument('--val-split', type=float, default=0.2,
                    help='Fraction of data used for validation')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    ap.add_argument('--output', help='Output config path')
    args = ap.parse_args()

    cfg = load_base_config(args.base_config)
    items = build_datalist(args.csv, args.data_dir, args.task)
    train_list, val_list = split_data(items, args.val_split, args.seed)

    mean, std = compute_stats(train_list)
    print(f"Computed mean: {mean:.4f} | std: {std:.4f}")

    cfg['train_list'] = train_list
    cfg['val_list'] = val_list
    cfg['val_split'] = args.val_split
    cfg['mean'] = mean
    cfg['std'] = std
    cfg['project'] = cfg.get('project', '')

    out_path = args.output
    if not out_path:
        out_path = os.path.join('configs', f'{args.task}_finetune.yaml')
    with open(out_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    print(f'Config written to {out_path}')


if __name__ == '__main__':
    main()
