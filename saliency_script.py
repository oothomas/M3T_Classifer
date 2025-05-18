#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary-edema assessment + IG-NoiseTunnel saliency for **144 � 144 � 144** cubes.
Compatible with the 14-May-2025 fine-tuned M3T-Edema checkpoint (epoch-5).

Key points
----------
? No BatchNorm ? GroupNorm / InstanceNorm conversion (matches training).
? Transformer depth kept at **12** (same as training).
? Positional-embedding length = 432 + 4 (CLS/SEP tokens) ? identical to training.
? Everything else (data transforms, registration, saliency maps) is unchanged.
"""

# --------------------------------------------------------------------------
# Standard & third-party imports
# --------------------------------------------------------------------------
import os, re, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, EnsureTyped
)

import torchvision.models as models
from einops import rearrange, repeat
from torch import Tensor

from captum.attr import IntegratedGradients, NoiseTunnel
import SimpleITK as sitk
import ants
from torch.cuda.amp import autocast


# --------------------------------------------------------------------------
# Checkpoint to load (fine-tuned model, epoch-5)
# --------------------------------------------------------------------------
CKPT_PATH = (
    "/home/oshane/Documents/ANTSpy/runs/20250516_215058_jw2mbiut/checkpoint_epoch025.pth"  # <<< CHANGED
)

# --------------------------------------------------------------------------
# Model code ? mirrors training-time implementation (BatchNorm retained)
# --------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -------------------- 1) 3-D CNN block --------------------
class CNN3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop_p: float = 0.10):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 5, padding=2)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout3d(drop_p)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.drop(x)


class MultiPlane_MultiSlice_Extract_Project(nn.Module):
    def __init__(self, out_channels: int, feat_drop_p=0.2):
        super().__init__()
        self.CNN_2D = models.resnet50(weights=None)
        self.CNN_2D.conv1 = nn.Conv2d(out_channels, 64, 7, 2, 3, bias=False)
        self.CNN_2D.fc    = nn.Identity()
        self.feat_drop = nn.Dropout(feat_drop_p)
        self.proj = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 256))
    def forward(self, t3d):
        B, C, D, H, W = t3d.shape
        cor = torch.cat(torch.split(t3d, 1, dim=2), dim=2)
        sag = torch.cat(torch.split(t3d, 1, dim=3), dim=3)
        axi = torch.cat(torch.split(t3d, 1, dim=4), dim=4)
        S = torch.cat(((cor * t3d).permute(0, 2, 1, 3, 4),
                       (sag * t3d).permute(0, 3, 1, 2, 4),
                       (axi * t3d).permute(0, 4, 1, 2, 3)), dim=1
                     ).view(-1, C, H, W)
        f2d = self.CNN_2D(S).view(B, -1, 2048)
        return self.proj(self.feat_drop(f2d))


# ===================== 3) Embedding Layer =====================
class EmbeddingLayer(nn.Module):
    """
    Adds positional + plane embeddings and token-dropout.

    Token bookkeeping is dynamic: any cube size works as long as
    total_tokens = 3 � slices_per_plane is passed at init.
    """
    def __init__(self, emb_size: int, total_tokens: int, token_drop_p: float):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.sep_token = nn.Parameter(torch.randn(1, 1, emb_size))

        # Plane-specific embeddings
        self.coronal_plane   = nn.Parameter(torch.randn(1, emb_size))
        self.sagittal_plane  = nn.Parameter(torch.randn(1, emb_size))
        self.axial_plane     = nn.Parameter(torch.randn(1, emb_size))

        # Positional embeddings: (CLS + 3 � plane + 3 � SEP) = total + 4
        self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))

        self.token_drop_p = token_drop_p

    def forward(self, tokens: Tensor):                           # (B, T, E)
        B, T, _ = tokens.shape
        tp = T // 3                                              # slices / plane

        cls = repeat(self.cls_token, "1 1 e -> b 1 e", b=B)
        sep = repeat(self.sep_token, "1 1 e -> b 1 e", b=B)

        x = torch.cat([
            cls,
            tokens[:, :tp, :],
            sep,
            tokens[:, tp:2*tp, :],
            sep,
            tokens[:, 2*tp:, :],
            sep
        ], dim=1)                                               # (B, T+4, E)

        # Plane-specific additive embeddings
        cor_end = 1 + tp + 1
        sag_end = cor_end + tp + 1
        x[:, :cor_end]  += self.coronal_plane
        x[:, cor_end:sag_end] += self.sagittal_plane
        x[:, sag_end:]  += self.axial_plane

        # Token-level dropout (training only)
        if self.training and self.token_drop_p > 0.0:
            keep = torch.rand(B, x.size(1), device=x.device) > self.token_drop_p
            keep[:, 0]  = True                                   # keep CLS
            keep[:, -1] = True                                   # keep last SEP
            x = x * keep.unsqueeze(-1)

        return x + self.positions


# -------------------- 4) Transformer encoder --------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, e: int = 256, h: int = 8, p: float = 0.10):
        super().__init__()
        self.e, self.h = e, h
        self.qkv  = nn.Linear(e, 3 * e)
        self.proj = nn.Linear(e, e)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), "b n (h d t) -> t b h n d", h=self.h, t=3)
        q, k, v = qkv
        att = torch.softmax((q @ k.transpose(-2, -1)) / (self.e ** 0.5), dim=-1)
        out = rearrange(self.drop(att) @ v, "b h n d -> b n (h d)")
        return self.proj(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x):   return x + self.fn(x)


class FeedForward(nn.Sequential):
    def __init__(self, e: int = 256, exp: int = 2, p: float = 0.10):
        super().__init__(nn.Linear(e, exp * e),
                         nn.GELU(),
                         nn.Dropout(p),
                         nn.Linear(exp * e, e))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, e: int = 256, p: float = 0.10, dp: float = 0.10):
        super().__init__()
        self.dp1 = DropPath(dp)
        self.dp2 = DropPath(dp)
        self.res1 = ResidualAdd(nn.Sequential(
            nn.LayerNorm(e),
            MultiHeadAttention(e, p=p),
            nn.Dropout(p)
        ))
        self.res2 = ResidualAdd(nn.Sequential(
            nn.LayerNorm(e),
            FeedForward(e, p=p),
            nn.Dropout(p)
        ))

    def forward(self, x):
        x = x + self.dp1(self.res1.fn(x))
        x = x + self.dp2(self.res2.fn(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 12, e: int = 256,
                 p: float = 0.10, dp: float = 0.10):
        super().__init__()
        dpr = np.linspace(0, dp, depth)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(e, p, dpr[i]) for i in range(depth)]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class EdemaHead(nn.Module):
    def __init__(self, e: int = 256, c: int = 2):
        super().__init__()
        self.lin = nn.Linear(e, c)
    def forward(self, x): return self.lin(x[:, 0])


class M3T_Edema(nn.Module):
    def __init__(self,
                 in_ch: int = 1, out_ch: int = 64, e: int = 256, depth: int = 12,
                 p_cnn3d: float = 0.10, p_feat: float = 0.20,
                 p_tok: float = 0.10, p_trans: float = 0.10, p_dp: float = 0.10):
        super().__init__()
        self.cnn3d   = CNN3DBlock(in_ch, out_ch, p_cnn3d)
        self.project = MultiPlane_MultiSlice_Extract_Project(out_ch, p_feat)
        self.embed   = EmbeddingLayer(e, total_tokens=432, token_drop_p=p_tok)
        self.trans   = TransformerEncoder(depth, e, p_trans, p_dp)
        self.head    = EdemaHead(e, 2)

    def forward(self, x):
        x = self.cnn3d(x)
        x = self.project(x)
        x = self.embed(x)
        x = self.trans(x)
        return self.head(x)


# --------------------------------------------------------------------------
# Dataset helper and utility functions (unchanged)
# --------------------------------------------------------------------------
class NRrdDataset(Dataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["src_path"] = sample["image_meta_dict"]["filename_or_obj"]
        return sample


def compute_accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def to_plain_tensor(x):
    return (x.as_tensor() if hasattr(x, "as_tensor") else
            torch.tensor(x.detach().cpu().numpy(), device=x.device,
                         dtype=x.dtype))


def normalize_map(a):
    m = np.mean(np.abs(a))
    return a / m if m > 0 else a


def _to_sitk(x):
    return x if isinstance(x, sitk.Image) else sitk.ReadImage(x)


def _save_vol_like(arr, ref, path):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.CopyInformation(_to_sitk(ref))
    sitk.WriteImage(img, str(path))


def _save_map_like(arr, ref, path):
    _save_vol_like(normalize_map(arr), ref, path)


def _unwrap_attr(obj):
    """Captum NoiseTunnel returns nested tuples; peel until Tensor."""
    while isinstance(obj, (tuple, list)):
        obj = obj[0]
    return obj


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------ File list & transforms --------------------------
    folder = "/home/oshane/Documents/masked_cubes144_labeled_nrrd"
    rx  = re.compile(r"(Scan_\d{4}).*?_(\d+)_([A-Za-z0-9]+)_(\w+)\.nrrd$")
    emap = {"FALSE":0,"False":0,"0":0,"TRUE":1,"True":1,"1":1}

    files = sorted(glob.glob(os.path.join(folder, "Scan_*_*.nrrd")))
    data  = [{"image":f,
              "label_edema": emap[rx.search(os.path.basename(f)).group(3)],
              "scan_id": rx.search(os.path.basename(f)).group(1)}
             for f in files if rx.search(os.path.basename(f))]

    np.random.seed(42); np.random.shuffle(data); split=int(0.8*len(data))

    tf = Compose([
        LoadImaged("image", image_only=False, reader="ITKReader"),
        EnsureChannelFirstd("image", strict_check=False),
        Orientationd("image", axcodes="SAR"),
        NormalizeIntensityd("image", subtrahend=1776.835584,
                                      divisor   =5603.538718),
        EnsureTyped("image", dtype=torch.float32, track_meta=True)
    ])

    train_ds, val_ds = NRrdDataset(data[:split], tf), NRrdDataset(data[split:], tf)
    train_loader = DataLoader(train_ds, 1, False)
    val_loader   = DataLoader(val_ds,   1, False)

    # ------------------ Model ------------------------------------------
    model = M3T_Edema().to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    # Forward wrapper for Present-minus-Absent
    def f_pma(x):
        with autocast(dtype=torch.float16):
            l = model(x); return (l[:,1]-l[:,0]).unsqueeze(1)

    ig = IntegratedGradients(f_pma); nt = NoiseTunnel(ig)
    IG_KW = dict(n_steps=16, method="gausslegendre", internal_batch_size=1)
    NT_KW = dict(nt_type="smoothgrad", nt_samples=8,
                 nt_samples_batch_size=1, stdevs=0.12,
                 return_convergence_delta=True)

    zero_bl = torch.zeros((1,1,144,144,144), device=device)
    air_bl  = torch.full_like(zero_bl, -0.6)
    baselines = [zero_bl, air_bl]

    # ------------------ Output dirs ------------------------------------
    os.makedirs("binary_normalized_explain_out_144_pretrained_e005",       exist_ok=True)   # <<< CHANGED
    os.makedirs("binary_normalized_contrastive_maps_144_pretrained_e005",  exist_ok=True)   # <<< CHANGED
    sev = ["Absent","Present"]

    # ------------------ Quick validation -------------------------------
    with torch.no_grad(), autocast(dtype=torch.float16):
        acc = [compute_accuracy(model(b["image"].half().to(device)),
                                b["label_edema"].to(device))
               for b in val_loader]
    print(f"\nValidation accuracy on held-out 20 %: {np.mean(acc):.3f}\n")

    # ------------------ Saliency loop ----------------------------------
    for b in tqdm(train_loader, desc="Saliency generation"):
        x   = to_plain_tensor(b["image"].to(device)).half().requires_grad_(True)
        sid = b["scan_id"][0]; src = b["src_path"][0]
        true_lbl = int(b["label_edema"][0])

        with autocast(dtype=torch.float16): logits = model(x)
        pred = int(logits.argmax())

        # --- proceed only if prediction is correct ---------------------
        if pred != true_lbl:
            continue

        # save raw volume once
        p_raw = f"binary_normalized_explain_out_144_pretrained_e005/scan_{sid}_volume_RAW.nrrd"
        if not os.path.exists(p_raw):
            _save_vol_like(x[0,0].detach().cpu().numpy(), src, p_raw)

        # Present-minus-Absent saliency (average over baselines)
        attr_pma = torch.mean(torch.stack([
            _unwrap_attr(nt.attribute(x, baselines=bl, **IG_KW, **NT_KW))
            for bl in baselines]), 0)

        raw_pma = attr_pma[0,0].detach().cpu().numpy()
        _save_map_like(raw_pma, src,
            f"binary_normalized_contrastive_maps_144_pretrained_e005/"
            f"scan_{sid}_{sev[true_lbl]}_PresentMinusAbsent_RAW.nrrd")

        print(f"? {sid}: correct ({sev[pred]}) ? map saved")
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
