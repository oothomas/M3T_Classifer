#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3-plane encoder + transformer head for embryonic-CT **edema** classification
on **144 ? 144 ? 144** cubes (Absent vs Present).

??  NEW IN THIS VERSION  ??
? Everything from the 128? build is untouched **except**:
  1.  Data directory now points at *masked_cubes144_labeled_nrrd*.
  2.  `wandb` run name and `config.resize` updated to **144**.
  3.  Token bookkeeping in **EmbeddingLayer** is made *dynamic* so it
     automatically handles 144 (or any other cube size) without hard-coding.
  4.  `EmbeddingLayer` is instantiated with `total_tokens = 432`
     (= 3 planes ? 144 slices), and its positional-embedding size expands
     accordingly.

No other architectural, training-schedule, or hyper-parameter changes have
been introduced.
"""

import os, re, glob, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, RandAffined, RandRotate90d,
    RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd,
    EnsureTyped, ToTensord, RandFlipd, RandGaussianNoised,
    RandGaussianSmoothd, RandCoarseDropoutd
)

import torchvision.models as models
from einops import rearrange, repeat
from torch import Tensor
from torchvision.models import ResNet50_Weights
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------ #
# NEW: path to the SSL checkpoint to import                         #
# ------------------------------------------------------------------ #
ssl_ckpt_path = "/home2/oshane/PRETRAINED_WEIGHTS/20250515_193447_zf67a4g5/ssl_ckpt_0310.pth"  # <<< CHANGED


# ------------------------------------------------------------------ #
# Utility: DropPath (stochastic depth) ? local fallback, no timm req.
# ------------------------------------------------------------------ #
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)          # broadcast mask
        random_tensor = keep_prob + torch.rand(shape,
                                               dtype=x.dtype, device=x.device)
        random_tensor.floor_()                                # 0 / 1
        return x.div(keep_prob) * random_tensor


# -------------------- 1) 3-D CNN block --------------------
class CNN3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 5, padding=2)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 5, padding=2)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout3d(drop_p)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.drop(x)


# -------------------- 2) Multi-plane 2-D CNN --------------------
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

    *Token bookkeeping is dynamic*: any cube size (e.g. 128?, 144?) works
    as long as `total_tokens = 3 ? slices_per_plane` is passed at init.
    """
    def __init__(self, emb_size: int, total_tokens: int, token_drop_p: float):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.sep_token = nn.Parameter(torch.randn(1, 1, emb_size))

        # Plane-specific embeddings
        self.coronal_plane   = nn.Parameter(torch.randn(1, emb_size))
        self.sagittal_plane  = nn.Parameter(torch.randn(1, emb_size))
        self.axial_plane     = nn.Parameter(torch.randn(1, emb_size))

        # Positional embeddings: (CLS + 3 planes + 3 SEP) = total + 4
        self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))

        self.token_drop_p = token_drop_p

    def forward(self, input_tensor: Tensor):                     # (B, T, E)
        B, T, _ = input_tensor.shape
        tp = T // 3                                              # slices/plane

        cls = repeat(self.cls_token, '() n e -> b n e', b=B)
        sep = repeat(self.sep_token, '() n e -> b n e', b=B)

        # Assemble [CLS | Cor | SEP | Sag | SEP | Ax | SEP]
        x = torch.cat([
            cls,
            input_tensor[:, :tp, :],
            sep,
            input_tensor[:, tp:2*tp, :],
            sep,
            input_tensor[:, 2*tp:, :],
            sep
        ], dim=1)                                               # (B, T+4, E)

        # Plane-specific additive embeddings
        cor_end = 1 + tp + 1
        sag_end = cor_end + tp + 1
        x[:, :cor_end]  += self.coronal_plane                    # CLS+Cor+SEP
        x[:, cor_end:sag_end] += self.sagittal_plane             # Sag+SEP
        x[:, sag_end:]  += self.axial_plane                      # Ax+SEP

        # Token-level dropout
        if self.training and self.token_drop_p > 0.0:
            keep = torch.rand(B, x.size(1), device=x.device) > self.token_drop_p
            keep[:, 0]  = True                                   # keep CLS
            keep[:, -1] = True                                   # keep last SEP
            x = x * keep.unsqueeze(-1)

        return x + self.positions                               # add position


# -------------------- 4) Transformer encoder --------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=256, heads=8, drop=0.1):
        super().__init__()
        self.emb_size, self.heads = emb_size, heads
        self.qkv  = nn.Linear(emb_size, emb_size * 3)
        self.proj = nn.Linear(emb_size, emb_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        qkv = rearrange(self.qkv(x),
                        'b n (h d qkv) -> qkv b h n d',
                        h=self.heads, qkv=3)
        q, k, v = qkv
        att = torch.softmax((q @ k.transpose(-2, -1)) /
                            (self.emb_size ** 0.5), dim=-1)
        out = rearrange(self.drop(att) @ v, 'b h n d -> b n (h d)')
        return self.proj(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class FeedForward(nn.Sequential):
    def __init__(self, emb_size, exp=2, drop=0.1):
        super().__init__(nn.Linear(emb_size, exp * emb_size),
                         nn.GELU(),
                         nn.Dropout(drop),
                         nn.Linear(exp * emb_size, emb_size))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=256, drop=0.1, drop_path=0.1, exp=2):
        super().__init__()
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.res1 = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size, drop=drop),
            nn.Dropout(drop)
        ))
        self.res2 = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForward(emb_size, exp, drop),
            nn.Dropout(drop)
        ))

    def forward(self, x):
        x = x + self.drop_path1(self.res1.fn(x))
        x = x + self.drop_path2(self.res2.fn(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth=10, emb_size=256, drop=0.1, drop_path=0.1):
        super().__init__()
        dpr = np.linspace(0, drop_path, depth)                   # progressive DP
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(emb_size, drop, dpr[i])
             for i in range(depth)]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# -------------------- 5) Binary head --------------------
class EdemaHead(nn.Module):
    def __init__(self, emb_size=256, classes=2):
        super().__init__()
        self.lin = nn.Linear(emb_size, classes)

    def forward(self, x):
        return self.lin(x[:, 0])                                 # take CLS token


# -------------------- 6) Full model --------------------
class M3T_Edema(nn.Module):
    def __init__(self,
                 in_ch=1,
                 out_ch=64,
                 emb_size=256,
                 depth=10,
                 p_cnn3d=0.1,
                 p_feat=0.2,
                 p_token=0.1,
                 p_transformer=0.1,
                 p_drop_path=0.1):
        super().__init__()
        self.cnn3d   = CNN3DBlock(in_ch, out_ch, drop_p=p_cnn3d)

        self.project = MultiPlane_MultiSlice_Extract_Project(out_ch, p_feat)

        # ------------- changed: total_tokens = 432 (3 ? 144) -------------
        self.embed   = EmbeddingLayer(emb_size, total_tokens=432,
                                      token_drop_p=p_token)
        # -----------------------------------------------------------------

        self.trans   = TransformerEncoder(depth, emb_size,
                                          drop=p_transformer,
                                          drop_path=p_drop_path)
        self.head    = EdemaHead(emb_size, 2)

    def forward(self, x):
        x = self.cnn3d(x)
        x = self.project(x)
        x = self.embed(x)
        x = self.trans(x)
        return self.head(x)


# -------------------- utils --------------------
def compute_accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


# -------------------- main --------------------
def main():
    wandb.init(project="Embryo_Edema_Classification_144_Masked_Pretrained",
               config=dict(
                   learning_rate      = 1e-4,
                   epochs             = 1000,
                   batch_size         = 3,          # adjust if GPU OOMs
                   resize             = (144, 144, 144),
                   out_channels       = 64,
                   emb_size           = 256,
                   depth              = 12,
                   num_classes        = 2,
                   p_cnn3d            = 0.10,
                   p_feat             = 0.20,
                   p_token            = 0.10,
                   p_transformer      = 0.10,
                   p_drop_path        = 0.10))
    config = wandb.config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{timestamp}_{wandb.run.id}")
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Custom Dataset retaining original NRRD path
    # ------------------------------------------------------------------
    class NRrdDataset(Dataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["src_path"] = sample["image_meta_dict"]["filename_or_obj"]
            return sample

    # ------------------------------------------------------------------
    # Build file-list (class-tagged NRRDs)
    # ------------------------------------------------------------------
    image_folder = "/home/oshane/Documents/masked_cubes144_labeled_nrrd"
    pattern      = os.path.join(image_folder, "Scan_*_*.nrrd")
    edema_map    = {"FALSE": 0, "False": 0, "0": 0,
                    "TRUE":  1, "True":  1, "1": 1}

    rx = re.compile(r"(Scan_\d{4}).*?_(\d+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)\.nrrd$")

    data_list = [dict(image=f,
                      label_edema=edema_map[rx.search(os.path.basename(f)).group(3)],
                      scan_id=rx.search(os.path.basename(f)).group(1))
                 for f in sorted(glob.glob(pattern))
                 if rx.search(os.path.basename(f))]

    np.random.seed(42)
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_data, val_data = data_list[:split], data_list[split:]

    # ------------------------------------------------------------------
    # Transform builder
    # ------------------------------------------------------------------
    def get_tf(mean=1776.835584, std=5603.538718):
        common = [
            LoadImaged(keys=["image"], image_only=False, reader="ITKReader"),
            EnsureChannelFirstd(keys=["image"], strict_check=False),
            Orientationd(keys=["image"], axcodes="SAR"),
            NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std)
        ]
        aug = [
            # RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),      # LR
            # RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),      # AP
            RandAffined(
                keys=["image"],
                prob=0.7,
                rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                translate_range=(4, 4, 4),
                scale_range=(0.05, 0.05, 0.05),
                mode=("bilinear",),
                padding_mode="border",
            ),
            # RandRotate90d(keys=["image"], prob=0.3, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.15,
                                sigma_x=(0.0, 1.0), sigma_y=(0.0, 1.0), sigma_z=(0.0, 1.0)),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.4)),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.3),
            RandCoarseDropoutd(
                keys=["image"],
                prob=0.2,
                holes=4,
                spatial_size=(16, 16, 16),
                max_holes=4,
                fill_value=0,
            ),
        ]
        final = [
            EnsureTyped(keys="image", dtype=torch.float32, track_meta=True),
            ToTensord(keys=["label_edema"])
        ]
        return Compose(common + aug + final), Compose(common + final)

    train_tf, val_tf = get_tf()

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(NRrdDataset(train_data, train_tf),
                              batch_size=config.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(NRrdDataset(val_data, val_tf),
                              batch_size=config.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Model & optimisation
    # ------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = M3T_Edema(1, config.out_channels, config.emb_size, config.depth,
                       config.p_cnn3d, config.p_feat, config.p_token,
                       config.p_transformer, config.p_drop_path).to(device)

    # -------------------- NEW: load SSL encoder weights ---------------- #
    if os.path.isfile(ssl_ckpt_path):
        ckpt = torch.load(ssl_ckpt_path, map_location="cpu")
        msg  = model.load_state_dict(ckpt["encoder"], strict=False)       # <<< CHANGED
        print(f"? SSL weights loaded ({len(msg.missing_keys)} params left random)")
    else:
        raise FileNotFoundError(f"SSL checkpoint not found: {ssl_ckpt_path}")
    # ------------------------------------------------------------------- #

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler    = GradScaler()

    counts = [0, 0]
    for s in train_data:
        counts[s["label_edema"]] += 1
    w = torch.tensor([sum(counts) / c for c in counts], device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    train_losses = []; val_losses = []; train_accs = []; val_accs = []

    for epoch in range(config.epochs):
        model.train()
        tloss = tacc = n = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{config.epochs}")
        for batch in pbar:
            imgs = batch["image"].to(device)
            labs = batch["label_edema"].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits = model(imgs)
                loss   = criterion(logits, labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            acc = compute_accuracy(logits, labs)
            tloss += loss.item() * labs.size(0)
            tacc  += acc * labs.size(0)
            n     += labs.size(0)
            pbar.set_postfix(loss=tloss/n, acc=tacc/n)

        train_losses.append(tloss / n)
        train_accs.append(tacc / n)

        model.eval()
        vloss = vacc = vn = 0
        with torch.no_grad(), autocast(dtype=torch.float16):
            for batch in tqdm(val_loader, desc="Val"):
                imgs = batch["image"].to(device)
                labs = batch["label_edema"].to(device).long()
                logits = model(imgs)
                loss   = criterion(logits, labs)
                acc    = compute_accuracy(logits, labs)
                vloss += loss.item() * labs.size(0)
                vacc  += acc * labs.size(0)
                vn    += labs.size(0)

        val_losses.append(vloss / vn)
        val_accs.append(vacc / vn)
        torch.cuda.empty_cache()

        wandb.log(dict(epoch=epoch + 1,
                       train_loss=train_losses[-1],
                       val_loss=val_losses[-1],
                       train_acc=train_accs[-1],
                       val_acc=val_accs[-1]))

        if (epoch + 1) % 5 == 0:
            ckpt = dict(epoch     = epoch + 1,
                        model     = model.state_dict(),
                        opt       = optimizer.state_dict(),
                        scheduler = scheduler.state_dict())
            torch.save(ckpt, os.path.join(
                run_dir, f"checkpoint_epoch{epoch + 1:03d}.pth"))

        print(f"Epoch {epoch+1}: "
              f"TrainLoss {train_losses[-1]:.4f} Acc {train_accs[-1]:.3f} | "
              f"ValLoss {val_losses[-1]:.4f} Acc {val_accs[-1]:.3f}")

        scheduler.step()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Curves
    # ------------------------------------------------------------------
    epochs = range(1, config.epochs + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(os.path.join(run_dir, "acc_curve.png"))
    plt.close()


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
