#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic Integrated?Gradients saliency for binary exencephaly
(fp16 version, IG n_steps=128, SmoothGrad ?=0.03)

Produces TWO saliency maps per correctly classified scan:
  ? ZERO?baseline map
  ? BLUR?baseline map (3?D average?blurred baseline)
"""
# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import os, re, glob, random, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, EnsureTyped
)
import torchvision.models as models
from einops import rearrange, repeat
from captum.attr import IntegratedGradients, NoiseTunnel
import torch.nn.functional as F                        # 3?D average blurring
# ==================== ADDED / CHANGED ====================
from torch.cuda.amp import autocast                    # enable fp16 autocast
# =========================================================

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Deterministic IG saliency for exencephaly models (fp16)")
parser.add_argument("--ckpt_dir", required=True,
                    help="Path to run directory (contains checkpoint_epochXXX.pth)")
parser.add_argument("--epoch", type=int, default=260,
                    help="Epoch number to load (default 260)")
parser.add_argument("--data_dir", default=
    "FL-SMALL/masked_cubes144_labeled_nrrd",
    help="Folder with NRRD cubes (default set to project path)")
parser.add_argument("--seed", type=int, default=42,
                    help="Global RNG seed (default 42)")
args = parser.parse_args()

# --------------------------------------------------------------------------
# Global determinism -------------------------------------------------------
SEED = args.seed
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

# --------------------------------------------------------------------------
# Checkpoint & output paths ------------------------------------------------
RUN_ID   = os.path.basename(os.path.abspath(args.ckpt_dir))
CKPT_PATH = os.path.join(args.ckpt_dir,
                         f"checkpoint_epoch{args.epoch:03d}.pth")
if not os.path.isfile(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

OUT_DIR = os.path.join("maps_small_updated", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Model definition  (unchanged)
# --------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__(); self.p = float(p)
    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x.div(keep) * rnd

class CNN3DBlock(nn.Module):
    def __init__(self, in_ch=1, out_ch=64, p=0.10):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 5, padding=2)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout3d(p)
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.drop(x)

class MultiPlane_MultiSlice_Extract_Project(nn.Module):
    def __init__(self, out_ch=64, p=0.20, emb_size=128):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.CNN_2D = models.resnet50(weights=weights)
        self.CNN_2D.conv1 = nn.Conv2d(out_ch, 64, 7, 2, 3, bias=False)
        self.CNN_2D.fc = nn.Identity()
        self.feat_drop = nn.Dropout(p)
        self.proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(),
                                  nn.Linear(512, emb_size))
    def forward(self, t3d):
        B, C, D, H, W = t3d.shape
        cor = torch.cat(torch.split(t3d, 1, 2), 2)
        sag = torch.cat(torch.split(t3d, 1, 3), 3)
        axi = torch.cat(torch.split(t3d, 1, 4), 4)
        S = torch.cat(((cor*t3d).permute(0,2,1,3,4),
                       (sag*t3d).permute(0,3,1,2,4),
                       (axi*t3d).permute(0,4,1,2,3)), 1).view(-1, C, H, W)
        f2d = self.CNN_2D(S).view(B, -1, 2048)
        return self.proj(self.feat_drop(f2d))

class EmbeddingLayer(nn.Module):
    def __init__(self, e=128, total_tokens=432, p_tok=0.10):
        super().__init__()
        self.cls_token      = nn.Parameter(torch.randn(1,1,e))
        self.sep_token      = nn.Parameter(torch.randn(1,1,e))
        self.coronal_plane  = nn.Parameter(torch.randn(1,e))
        self.sagittal_plane = nn.Parameter(torch.randn(1,e))
        self.axial_plane    = nn.Parameter(torch.randn(1,e))
        self.positions      = nn.Parameter(torch.randn(total_tokens+4, e))
        self.p_tok = p_tok
    def forward(self, t):
        B, T, _ = t.shape
        tp = T // 3
        cls = repeat(self.cls_token, '1 1 e -> b 1 e', b=B)
        sep = repeat(self.sep_token, '1 1 e -> b 1 e', b=B)
        x = torch.cat([cls,
                       t[:, :tp], sep,
                       t[:, tp:2*tp], sep,
                       t[:, 2*tp:], sep], 1)
        cor_end = 1 + tp + 1
        sag_end = cor_end + tp + 1
        x[:,      :cor_end]   += self.coronal_plane
        x[:, cor_end:sag_end] += self.sagittal_plane
        x[:, sag_end:]        += self.axial_plane
        if self.training and self.p_tok > 0:
            keep = torch.rand(B, x.size(1), device=x.device) > self.p_tok
            keep[:, 0]  = True
            keep[:, -1] = True
            x = x * keep.unsqueeze(-1)
        return x + self.positions

class MultiHeadAttention(nn.Module):
    def __init__(self, e=128, h=4, p=0.10):
        super().__init__()
        self.e, self.h = e, h
        self.qkv  = nn.Linear(e, 3*e)
        self.proj = nn.Linear(e, e)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        qkv = rearrange(self.qkv(x),
                        'b n (h d qkv) -> qkv b h n d',
                        h=self.h, qkv=3)
        q, k, v = qkv
        att = torch.softmax((q @ k.transpose(-2, -1)) / (self.e**0.5), -1)
        out = rearrange(self.drop(att) @ v, 'b h n d -> b n (h d)')
        return self.proj(out)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    def forward(self, x): return x + self.fn(x)

class FeedForward(nn.Sequential):
    def __init__(self, e=128, exp=2, p=0.10):
        super().__init__(nn.Linear(e, exp*e), nn.GELU(), nn.Dropout(p),
                         nn.Linear(exp*e, e))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, e=128, p=0.10, dp=0.10):
        super().__init__()
        self.dp1 = DropPath(dp); self.dp2 = DropPath(dp)
        self.res1 = ResidualAdd(nn.Sequential(nn.LayerNorm(e),
                                              MultiHeadAttention(e, p=p),
                                              nn.Dropout(p)))
        self.res2 = ResidualAdd(nn.Sequential(nn.LayerNorm(e),
                                              FeedForward(e, p=p),
                                              nn.Dropout(p)))
    def forward(self, x):
        x = x + self.dp1(self.res1.fn(x))
        x = x + self.dp2(self.res2.fn(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth=4, e=128, p=0.10, dp=0.10):
        super().__init__()
        import numpy as _np
        dpr = _np.linspace(0, dp, depth)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(e, p, dpr[i])
                                     for i in range(depth)])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class ExencephalyHead(nn.Module):
    def __init__(self, e=128):
        super().__init__()
        self.lin = nn.Linear(e, 2)
    def forward(self, x):
        return self.lin(x[:, 0])

class M3T_Exencephaly(nn.Module):
    def __init__(self, in_ch=1, out_ch=64, e=128, depth=4,
                 p_cnn3d=0.10, p_feat=0.20, p_tok=0.10,
                 p_trans=0.10, p_dp=0.10):
        super().__init__()
        self.cnn3d  = CNN3DBlock(in_ch, out_ch, p_cnn3d)
        self.project = MultiPlane_MultiSlice_Extract_Project(out_ch, p_feat, e)
        self.embed  = EmbeddingLayer(e, total_tokens=432, p_tok=p_tok)
        self.trans  = TransformerEncoder(depth, e, p_trans, p_dp)
        self.head   = ExencephalyHead(e)
    def forward(self, x):
        x = self.cnn3d(x)
        x = self.project(x)
        x = self.embed(x)
        x = self.trans(x)
        return self.head(x)

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
class NRrdDataset(Dataset):
    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        s["src_path"] = s["image_meta_dict"]["filename_or_obj"]
        return s

def compute_accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

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
    img = sitk.GetImageFromArray(normalize_map(arr).astype(np.float32))
    img.CopyInformation(_to_sitk(ref))
    sitk.WriteImage(img, str(path))

def _unwrap_attr(o):
    while isinstance(o, (tuple, list)):
        o = o[0]
    return o

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------- dataset ---------------------------------------------------
    rx = re.compile(r"(Scan_\d{4}).*?_(\d+)_([A-Za-z0-9]+)_(\w+)\.nrrd$")
    emap = {"FALSE":0,"False":0,"0":0,"TRUE":1,"True":1,"1":1}
    files = sorted(glob.glob(os.path.join(args.data_dir, "Scan_*_*.nrrd")))
    data = [{"image": f,
             "label_exencephaly": emap[rx.search(os.path.basename(f)).group(4)],
             "scan_id": rx.search(os.path.basename(f)).group(1)}
            for f in files if rx.search(os.path.basename(f))]
    tf = Compose([
        LoadImaged("image", image_only=False, reader="ITKReader"),
        EnsureChannelFirstd("image", strict_check=False),
        Orientationd("image", axcodes="SAR"),
        NormalizeIntensityd("image", subtrahend=1755.992392,
                                     divisor   =5557.190527),
        EnsureTyped("image", dtype=torch.float32, track_meta=True)
    ])
    ds = NRrdDataset(data, tf)
    loader = DataLoader(ds, 1, False)

    # ---------- model -----------------------------------------------------
    model = M3T_Exencephaly().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    # ---------- Captum ----------------------------------------------------
    def f_pma(x):
        """Present?minus?Absent logit difference (fp16)."""
        with autocast(dtype=torch.float16):                       # fp16
            logits = model(x)
            return (logits[:, 1] - logits[:, 0]).unsqueeze(1)

    ig = IntegratedGradients(f_pma)
    nt = NoiseTunnel(ig)
    IG_KW = dict(n_steps=64, method="gausslegendre", internal_batch_size=4)
    NT_KW = dict(nt_type="smoothgrad", nt_samples=12,
                 nt_samples_batch_size=4, stdevs=0.03,
                 return_convergence_delta=True)

    zero_bl = torch.zeros((1, 1, 144, 144, 144), device=device, dtype=torch.float16)  # fp16 baseline
    sev = ["Absent", "Present"]
    correct_ids = []

    # ---------- quick accuracy -------------------------------------------
    with torch.no_grad(), autocast(dtype=torch.float16):          # fp16
        acc = [compute_accuracy(model(b["image"].half().to(device)),
                                b["label_exencephaly"].to(device))
               for b in loader]
    print(f"[{RUN_ID}] Dataset accuracy (all scans): {np.mean(acc):.3f}")

    # ---------- saliency loop --------------------------------------------
    for idx, b in enumerate(tqdm(loader, desc=f"Saliency ({RUN_ID})")):
        x = b["image"].to(device).half().requires_grad_(True)     # fp16
        sid = b["scan_id"][0]
        src = b["src_path"][0]
        true_lbl = int(b["label_exencephaly"][0])

        with autocast(dtype=torch.float16):                       # fp16
            logits = model(x)
        pred = int(logits.argmax())

        if pred != true_lbl:      # only analyse correct predictions
            continue
        correct_ids.append(sid)

        # save raw volume once per scan
        p_raw = os.path.join(OUT_DIR, f"scan_{sid}_volume_RAW.nrrd")
        if not os.path.exists(p_raw):
            _save_vol_like(x[0, 0].detach().cpu().numpy(), src, p_raw)

        # Construct blurred baseline (15?voxel avg pool, fp16)
        blur_bl = F.avg_pool3d(x.detach(), kernel_size=15, stride=1, padding=7)

        # ---------- ZERO baseline attribution -------------------
        torch.manual_seed(SEED + idx)
        attr_zero = _unwrap_attr(nt.attribute(x, baselines=zero_bl,
                                              **IG_KW, **NT_KW))
        fname_zero = f"{sid}_{sev[true_lbl]}_PresentMinusAbsent_ZERO_RAW.nrrd"
        _save_map_like(attr_zero[0, 0].cpu().numpy(), src,
                       os.path.join(OUT_DIR, fname_zero))

        # ---------- BLUR baseline attribution -------------------
        torch.manual_seed(SEED + idx)
        attr_blur = _unwrap_attr(nt.attribute(x, baselines=blur_bl,
                                              **IG_KW, **NT_KW))
        fname_blur = f"{sid}_{sev[true_lbl]}_PresentMinusAbsent_BLUR_RAW.nrrd"
        _save_map_like(attr_blur[0, 0].cpu().numpy(), src,
                       os.path.join(OUT_DIR, fname_blur))

        torch.cuda.empty_cache()

    # ---------- save list of correct ids ---------------------------------
    with open(os.path.join(OUT_DIR, "correct_ids.txt"), "w") as f:
        f.write("\n".join(correct_ids))
    print(f"[{RUN_ID}] Done. Correctly classified: {len(correct_ids)}/{len(ds)}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()