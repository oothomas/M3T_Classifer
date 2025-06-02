"""Saliency map generation using Captum."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, NoiseTunnel
from torch.cuda.amp import autocast
from tqdm import tqdm

from models.classifier import M3T_Edema
from data.transforms import build_transforms
from data.dataset import NRRDDataset


def generate_maps(cfg: dict) -> None:
    """Generate and save saliency maps for a dataset."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = build_transforms(cfg['mean'], cfg['std'])[0]
    loader = DataLoader(NRRDDataset(cfg['data_list'], tf), batch_size=1, shuffle=False)

    model = M3T_Edema(cfg).to(device)
    ckpt_path = cfg['ckpt']
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(cfg.get('ckpt_dir', 'checkpoints'), ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    def f_pma(x):
        """Forward function for Captum that returns the logit difference."""

        with autocast(dtype=torch.float16):
            logits = model(x)
            return (logits[:, 1] - logits[:, 0]).unsqueeze(1)

    ig = IntegratedGradients(f_pma); nt = NoiseTunnel(ig)
    for b in tqdm(loader, desc='Saliency'):
        x = b['image'].to(device).half().requires_grad_(True)
        sid = b.get('scan_id',["0"])[0]
        attr = nt.attribute(x, baselines=torch.zeros_like(x), n_steps=16, nt_type='smoothgrad', nt_samples=8)
        arr = attr[0,0].detach().cpu().numpy()
        os.makedirs(cfg['out_dir'], exist_ok=True)
        np.save(os.path.join(cfg['out_dir'], f'scan_{sid}_saliency.npy'), arr)
