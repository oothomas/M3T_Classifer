#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a deterministic 3-plane CNN-ViT for binary exencephaly classification
on 144³ µCT embryo volumes.

Key features:
    - Uses focal loss (gamma=2) with DropPath disabled by default.
    - Reduced model capacity (embedding size 128, 4 heads, depth 4).
    - ResNet-50 slice encoder initialised with ImageNet weights.
    - Slice encoder projects to 128 dimensions to match the transformer
      embedding layer.
"""
# ------------------------------------------------------------------- imports
import os, re, glob, random, warnings, collections, argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai.utils import set_determinism
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, RandAffined, RandGaussianNoised,
    RandGaussianSmoothd, RandAdjustContrastd, RandScaleIntensityd,
    RandShiftIntensityd, RandCoarseDropoutd, EnsureTyped, ToTensord
)
import torchvision.models as models  # Requires torchvision >= 0.15
from einops import rearrange, repeat
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import wandb
# ------------------------------------------------------------------- CLI / seed
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int,
                    help="global RNG seed (overrides $SEED)")
args, _ = parser.parse_known_args()

SEED = args.seed if args.seed is not None else int(os.getenv("SEED", "0"))
print(f"[INFO] Using deterministic seed: {SEED}")

# deterministic flags
set_determinism(seed=SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
warnings.filterwarnings("ignore", "torch.*deterministic.*")
# ------------------------------------------------------------------- model defs
class DropPath(nn.Module):
    """Stochastic depth regularisation layer."""

    def __init__(self, p: float = 0.0):
        """Initialise the DropPath module.

        Args:
            p (float, optional): Drop probability. Defaults to ``0.0``.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """Apply drop path to the input tensor."""
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x.div(keep) * rnd

class CNN3DBlock(nn.Module):
    """Simple two-layer 3D convolutional block."""

    def __init__(self, in_ch=1, out_ch=64, p=0.10):
        """Initialise the 3D CNN block.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            p (float): Dropout probability.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 5, padding=2)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.drop = nn.Dropout3d(p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass through the block."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.drop(x)

class MultiPlane_MultiSlice_Extract_Project(nn.Module):
    """Extract features from slice mosaics using a 2D ResNet-50."""

    def __init__(self, out_ch=64, p=0.20, emb_size=128):
        """Initialise the slice encoder and projection head.

        Args:
            out_ch (int): Number of channels from the 3D CNN block.
            p (float): Dropout probability applied to features.
            emb_size (int): Transformer embedding dimension.
        """
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.CNN_2D = models.resnet50(weights=weights)
        self.CNN_2D.conv1 = nn.Conv2d(out_ch, 64, 7, 2, 3, bias=False)
        self.CNN_2D.fc = nn.Identity()
        self.feat_drop = nn.Dropout(p)
        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, emb_size),
        )

    def forward(self, t3d):
        """Encode tri-planar slices and project to embeddings."""
        B, C, D, H, W = t3d.shape
        cor = torch.cat(torch.split(t3d, 1, 2), 2)
        sag = torch.cat(torch.split(t3d, 1, 3), 3)
        axi = torch.cat(torch.split(t3d, 1, 4), 4)
        S = torch.cat(
            (
                (cor * t3d).permute(0, 2, 1, 3, 4),
                (sag * t3d).permute(0, 3, 1, 2, 4),
                (axi * t3d).permute(0, 4, 1, 2, 3),
            ),
            1,
        ).view(-1, C, H, W)
        f2d = self.CNN_2D(S).view(B, -1, 2048)
        return self.proj(self.feat_drop(f2d))

class EmbeddingLayer(nn.Module):
    """Token and positional embeddings for tri-planar inputs."""

    def __init__(self, e=128, total_tokens=432, p_tok=0.10):
        """Initialise embedding tensors.

        Args:
            e (int): Embedding dimension.
            total_tokens (int): Total number of tokens per sample.
            p_tok (float): Token dropout probability.
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, e))
        self.sep_token = nn.Parameter(torch.randn(1, 1, e))
        self.coronal_plane = nn.Parameter(torch.randn(1, e))
        self.sagittal_plane = nn.Parameter(torch.randn(1, e))
        self.axial_plane = nn.Parameter(torch.randn(1, e))
        self.positions = nn.Parameter(torch.randn(total_tokens + 4, e))
        self.p_tok = p_tok

    def forward(self, t):
        """Construct token sequence with plane and positional embeddings."""
        B, T, _ = t.shape
        tp = T // 3
        cls = repeat(self.cls_token, '1 1 e -> b 1 e', b=B)
        sep = repeat(self.sep_token, '1 1 e -> b 1 e', b=B)
        x = torch.cat(
            [
                cls,
                t[:, :tp],
                sep,
                t[:, tp : 2 * tp],
                sep,
                t[:, 2 * tp :],
                sep,
            ],
            1,
        )
        cor_end = 1 + tp + 1
        sag_end = cor_end + tp + 1
        x[:, :cor_end] += self.coronal_plane
        x[:, cor_end:sag_end] += self.sagittal_plane
        x[:, sag_end:] += self.axial_plane
        if self.training and self.p_tok > 0:
            keep = torch.rand(B, x.size(1), device=x.device) > self.p_tok
            keep[:, 0] = True
            keep[:, -1] = True
            x = x * keep.unsqueeze(-1)
        return x + self.positions

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with 4 heads of 32 dimensions."""

    def __init__(self, e=128, h=4, p=0.10):
        """Initialise the attention layer."""
        super().__init__()
        self.e, self.h = e, h
        self.qkv = nn.Linear(e, 3 * e)
        self.proj = nn.Linear(e, e)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Apply self-attention to the input tensor."""
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> qkv b h n d', h=self.h, qkv=3)
        q, k, v = qkv
        att = torch.softmax((q @ k.transpose(-2, -1)) / (self.e ** 0.5), -1)
        out = rearrange(self.drop(att) @ v, 'b h n d -> b n (h d)')
        return self.proj(out)

class ResidualAdd(nn.Module):
    """Add a residual connection around a given function."""

    def __init__(self, fn):
        """Store the function to be wrapped."""
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """Apply the function and add the input."""
        return x + self.fn(x)

class FeedForward(nn.Sequential):
    """Feed-forward network used within the transformer encoder."""

    def __init__(self, e=128, exp=2, p=0.10):
        """Initialise the feed-forward layers."""
        super().__init__(
            nn.Linear(e, exp * e),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(exp * e, e),
        )

class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, e=128, p=0.10, dp=0.0):
        """Initialise the encoder block."""
        super().__init__()
        self.dp1 = DropPath(dp)
        self.dp2 = DropPath(dp)
        self.res1 = ResidualAdd(
            nn.Sequential(nn.LayerNorm(e), MultiHeadAttention(e, p=p), nn.Dropout(p))
        )
        self.res2 = ResidualAdd(
            nn.Sequential(nn.LayerNorm(e), FeedForward(e, p=p), nn.Dropout(p))
        )

    def forward(self, x):
        """Run the encoder block."""
        x = x + self.dp1(self.res1.fn(x))
        x = x + self.dp2(self.res2.fn(x))
        return x

class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""

    def __init__(self, depth=4, e=128, p=0.10, dp=0.0):
        """Initialise the encoder module."""
        super().__init__()
        dpr = np.zeros(depth)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(e, p, dpr[i]) for i in range(depth)]
        )

    def forward(self, x):
        """Apply each block sequentially."""
        for blk in self.blocks:
            x = blk(x)
        return x

class ExencephalyHead(nn.Module):
    """Linear classification head for exencephaly prediction."""

    def __init__(self, e=128):
        """Initialise the classifier."""
        super().__init__()
        self.lin = nn.Linear(e, 2)

    def forward(self, x):
        """Return logits derived from the class token."""
        return self.lin(x[:, 0])

class M3T_Exencephaly(nn.Module):
    """Full model combining CNN and transformer components."""

    def __init__(
        self,
        in_ch=1,
        out_ch=64,
        e=128,
        depth=4,
        p_cnn3d=0.10,
        p_feat=0.20,
        p_tok=0.10,
        p_trans=0.10,
        p_dp=0.0,
    ):
        """Initialise the model architecture."""
        super().__init__()
        self.cnn3d = CNN3DBlock(in_ch, out_ch, p_cnn3d)
        self.project = MultiPlane_MultiSlice_Extract_Project(out_ch, p_feat, e)
        self.embed = EmbeddingLayer(e, total_tokens=432, p_tok=p_tok)
        self.trans = TransformerEncoder(depth, e, p_trans, p_dp)
        self.head = ExencephalyHead(e)

    def forward(self, x):
        """Run the full model on the input volume."""
        x = self.cnn3d(x)
        x = self.project(x)
        x = self.embed(x)
        x = self.trans(x)
        return self.head(x)
# ================================================================ FOCAL LOSS
class FocalLoss(nn.Module):
    """Weighted multi-class focal loss with fixed gamma."""

    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        """Initialise the focal loss.

        Args:
            weight (Tensor, optional): Class weighting tensor.
            gamma (float): Focusing parameter.
            reduction (str): Reduction mode (``"mean"`` or ``"sum"``).
        """
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        """Compute the focal loss."""
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
# ================================================================ helpers
class NRrdDataset(Dataset):
    """Dataset wrapper that retains source file paths."""

    def __getitem__(self, idx):
        """Return a sample with its source path attached."""
        s = super().__getitem__(idx)
        s["src_path"] = s["image_meta_dict"]["filename_or_obj"]
        return s


def compute_accuracy(logits, labels):
    """Compute classification accuracy."""
    return (logits.argmax(1) == labels).float().mean().item()
# ================================================================ main
def main():
    """Execute the training pipeline."""
    wandb.init(
        project="Embryo_Exencephaly_Classification_144_Masked_BALANCED",
        config=dict(
            learning_rate=1e-4,
            epochs=1000,
            batch_size=6,
            resize=(144, 144, 144),
            out_channels=64,
            emb_size=128,
            depth=4,
            p_cnn3d=0.10,
            p_feat=0.20,
            p_token=0.10,
            p_transformer=0.10,
            p_drop_path=0.0,
            seed=SEED,
        ),
    )
    cfg = wandb.config
    run_dir = os.path.join("runs", f"{datetime.now():%Y%m%d_%H%M%S}_{wandb.run.id}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "seed.txt"), "w") as f:
        f.write(str(SEED))

    # ------------------------- dataset file list -------------------
    folder = "masked_cubes144_labeled_nrrd"
    rx = re.compile(r"(Scan_\d{4}).*?_(\d+)_([A-Za-z0-9]+)_(\w+)\.nrrd$")
    emap = {"FALSE":0,"False":0,"0":0,"TRUE":1,"True":1,"1":1}
    raw = []
    for f in sorted(glob.glob(os.path.join(folder, "Scan_*_*.nrrd"))):
        m = rx.search(os.path.basename(f))
        if m:
            raw.append(dict(image=f,
                            label_exencephaly=emap[m.group(4)],
                            scan_id=m.group(1)))
    if not raw:
        raise RuntimeError("No NRRDs found.")

    # ------------------------- train/val split ---------------------
    splits_dir = "/data/hps/home/othoma/magalab/user/othoma/Gli2_classifier/splits"
    os.makedirs(splits_dir, exist_ok=True)
    tr_file  = os.path.join(splits_dir, f"exencephaly_train_idx_seed{SEED}.npy")
    val_file = os.path.join(splits_dir, f"exencephaly_val_idx_seed{SEED}.npy")

    if os.path.exists(tr_file) and os.path.exists(val_file):
        tr_idx = np.load(tr_file); val_idx = np.load(val_file)
        print(f"[INFO] Reusing cached split with {len(tr_idx)} train / {len(val_idx)} val.")
    else:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        y  = [r["label_exencephaly"] for r in raw]
        gp = [r["scan_id"]          for r in raw]
        tr_idx, val_idx = next(sgkf.split(raw, y, gp))
        np.save(tr_file, np.array(tr_idx))
        np.save(val_file, np.array(val_idx))
        print(f"[INFO] Saved new split indices to {splits_dir}.")

    train_data=[raw[i] for i in tr_idx]; val_data=[raw[i] for i in val_idx]

    # ------------------------- transforms --------------------------
    def get_tf(mean=1755.992392, std=5557.190527):
        """Create training and validation transforms."""
        common=[
            LoadImaged("image", image_only=False, reader="ITKReader"),
            EnsureChannelFirstd("image", strict_check=False),
            Orientationd("image", axcodes="SAR"),
            NormalizeIntensityd("image", subtrahend=mean, divisor=std),
        ]
        aug=[
            RandAffined("image", prob=0.7,
                        rotate_range=(np.pi/18,)*3,
                        translate_range=(4,4,4),
                        scale_range=(0.05,0.05,0.05),
                        mode=("bilinear",), padding_mode="border"),
            RandGaussianNoised("image", prob=0.25, mean=0.0, std=0.01),
            RandGaussianSmoothd("image", prob=0.15,
                                sigma_x=(0.0,1.0), sigma_y=(0.0,1.0), sigma_z=(0.0,1.0)),
            RandAdjustContrastd("image", prob=0.3, gamma=(0.7,1.4)),
            RandScaleIntensityd("image", factors=0.15, prob=0.3),
            RandShiftIntensityd("image", offsets=0.10, prob=0.3),
            RandCoarseDropoutd("image", prob=0.2, holes=4,
                               spatial_size=(16,16,16), max_holes=4,
                               fill_value=0),
        ]
        final=[EnsureTyped("image", dtype=torch.float32, track_meta=True),
               ToTensord("label_exencephaly")]
        return Compose(common+aug+final), Compose(common+final)
    train_tf,val_tf=get_tf()

    # ------------------------- datasets & loaders ------------------
    train_ds=NRrdDataset(train_data,train_tf); val_ds=NRrdDataset(val_data,val_tf)
    cls_freq=collections.Counter(r["label_exencephaly"] for r in train_data)
    w0,w1=1.0/cls_freq[0],1.0/cls_freq[1]
    sample_weights=[w1 if r["label_exencephaly"]==1 else w0 for r in train_data]

    g=torch.Generator(); g.manual_seed(SEED)
    sampler=WeightedRandomSampler(sample_weights, len(train_data), True, g)
    def worker_init_fn(worker_id):
        """Ensure each worker has a distinct, deterministic seed."""
        worker_seed = SEED + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader=DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn, generator=g)
    val_loader  =DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn, generator=g)

    # ------------------------- model & optim -----------------------
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=M3T_Exencephaly(1, cfg.out_channels, cfg.emb_size, cfg.depth,
                          cfg.p_cnn3d, cfg.p_feat, cfg.p_token,
                          cfg.p_transformer, cfg.p_drop_path).to(device)
    print("[INFO] ResNet-50 branch initialised from ImageNet.")

    optimiser=optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler=CosineAnnealingLR(optimiser, T_max=cfg.epochs)
    scaler   =GradScaler()

    counts=[cls_freq[0], cls_freq[1]]
    ce_w=torch.tensor([sum(counts)/c for c in counts], device=device)
    criterion=FocalLoss(weight=ce_w, gamma=2.0)

    # ------------------------- training loop -----------------------
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(cfg.epochs):
        model.train(); tl=ta=n=0
        pbar=tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for b in pbar:
            imgs=b["image"].to(device); labs=b["label_exencephaly"].to(device).long()
            optimiser.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits=model(imgs); loss=criterion(logits,labs)
            scaler.scale(loss).backward(); scaler.step(optimiser); scaler.update()
            acc=compute_accuracy(logits,labs)
            tl+=loss.item()*labs.size(0); ta+=acc*labs.size(0); n+=labs.size(0)
            pbar.set_postfix(loss=f"{tl/n:.4f}", acc=f"{ta/n:.3f}")
        train_loss.append(tl/n); train_acc.append(ta/n)

        # ------------------------- validation ----------------------
        model.eval(); vl=va=vn=0
        with torch.no_grad(), autocast(dtype=torch.float16):
            for b in tqdm(val_loader, desc="Val"):
                imgs=b["image"].to(device); labs=b["label_exencephaly"].to(device).long()
                logits=model(imgs); loss=criterion(logits,labs)
                acc=compute_accuracy(logits,labs)
                vl+=loss.item()*labs.size(0); va+=acc*labs.size(0); vn+=labs.size(0)
        val_loss.append(vl/vn); val_acc.append(va/vn)

        wandb.log(dict(epoch=epoch+1,
                       train_loss=train_loss[-1], val_loss=val_loss[-1],
                       train_acc=train_acc[-1], val_acc=val_acc[-1]))

        if (epoch+1) % 5 == 0:
            torch.save(dict(epoch=epoch+1, model=model.state_dict(),
                            optimiser=optimiser.state_dict(),
                            scheduler=scheduler.state_dict()),
                       os.path.join(run_dir, f"checkpoint_epoch{epoch+1:03d}.pth"))

        print(f"Epoch {epoch+1:04d} | Train {train_loss[-1]:.4f}/{train_acc[-1]:.3f} "
              f"|| Val {val_loss[-1]:.4f}/{val_acc[-1]:.3f}")
        scheduler.step(); torch.cuda.empty_cache()

    # --------------------------- curves ---------------------------
    eps=range(1, cfg.epochs+1)
    plt.figure(); plt.plot(eps, train_loss, label="Train"); plt.plot(eps, val_loss, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(os.path.join(run_dir, "loss_curve.png")); plt.close()

    plt.figure(); plt.plot(eps, train_acc, label="Train"); plt.plot(eps, val_acc, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(os.path.join(run_dir, "acc_curve.png")); plt.close()
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
