import argparse
import os
import glob
import random
import yaml
import pandas as pd


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
    random.Random(seed).shuffle(items)
    val_count = int(len(items) * val_split)
    val_list = items[:val_count]
    train_list = items[val_count:]
    return train_list, val_list


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

    cfg['train_list'] = train_list
    cfg['val_list'] = val_list
    cfg['val_split'] = args.val_split
    cfg['project'] = cfg.get('project', '')

    out_path = args.output
    if not out_path:
        out_path = os.path.join('configs', f'{args.task}_finetune.yaml')
    with open(out_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    print(f'Config written to {out_path}')


if __name__ == '__main__':
    main()
