#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Self-Supervised Pre-training for 3-Plane CNN + Transformer Encoder
=======================================================================

* BYOL contrastive alignment (no negatives, momentum target → cosine-ramped)
* Masked-voxel reconstruction (MSE on masked voxels)
* 3-axis 4-angle rotation prediction (12-class head)
* Binary slice-order (jigsaw) prediction head

Changes in this version (relative to your original v0)
------------------------------------------------------
✓ cosine-ramped BYOL τ (0.99 → 0.999)  
✓ mask_ratio 0.35  
✓ λ_contrast 2.0 , λ_recon 0.5  
✓ weight-decay 3 × 10⁻⁶  
✓ AdamW base-LR 6 × 10⁻⁴ with 10-epoch warm-up → cosine  
✓ CNN / projector dropouts 0.05 ; DropPath 0.05  
✓ Heavy RandCoarseDropout **only on view-2**  
✓ Rotation range ±5°  
✓ Batch size 12  
✓ wandb logs *every* loss component, LR and τ

ℹ The “gamma-flip” intensity inversion proposed earlier required  
`RandInvertIntensityd`, which is not present in your MONAI build.  
To keep the script fully compatible we have **removed that single line**;  
all other augmentations remain unchanged.
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import os, glob, math, copy, random
from datetime import datetime
from typing   import Tuple, List

import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import wandb

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, RandFlipd, RandAffined, RandAdjustContrastd,
    RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised,
    RandGaussianSmoothd, RandCoarseDropoutd, EnsureTyped
)

import torchvision.models as models
from einops import rearrange, repeat
from torch import Tensor
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------
# 1.  Core encoder blocks
# ---------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__(); self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype,
                                               device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class CNN3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_p=0.05):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 5, padding=2)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 5, padding=2)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout3d(drop_p)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(self.relu(self.bn2(self.conv2(x))))
        return x                                              # (B,C,D,H,W)

class MultiPlane_MultiSlice_Extract_Project(nn.Module):
    def __init__(self, out_channels: int, feat_drop_p=0.05):
        super().__init__()
        self.CNN_2D = models.resnet50(weights=None)
        self.CNN_2D.conv1 = nn.Conv2d(out_channels, 64, 7, 2, 3, bias=False)
        self.CNN_2D.fc    = nn.Identity()
        self.feat_drop = nn.Dropout(feat_drop_p)
        self.proj = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512, 256))
    def forward(self, t3d):
        B, C, D, H, W = t3d.shape
        cor = torch.cat(torch.split(t3d, 1, dim=2), dim=2)
        sag = torch.cat(torch.split(t3d, 1, dim=3), dim=3)
        axi = torch.cat(torch.split(t3d, 1, dim=4), dim=4)
        S = torch.cat(((cor * t3d).permute(0, 2, 1, 3, 4),
                       (sag * t3d).permute(0, 3, 1, 2, 4),
                       (axi * t3d).permute(0, 4, 1, 2, 3)), dim=1
                     ).reshape(-1, C, H, W)
        f2d = self.CNN_2D(S).view(B, -1, 2048)
        return self.proj(self.feat_drop(f2d))                 # (B,T,256)

class EmbeddingLayer(nn.Module):
    def __init__(self, emb_size: int, total_tokens: int, token_drop_p: float):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.sep_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.coronal_plane  = nn.Parameter(torch.randn(1, emb_size))
        self.sagittal_plane = nn.Parameter(torch.randn(1, emb_size))
        self.axial_plane    = nn.Parameter(torch.randn(1, emb_size))
        self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))
        self.token_drop_p = token_drop_p
    def forward(self, x: Tensor):
        B, T, _ = x.shape; tp = T // 3
        cls = repeat(self.cls_token, '() n e -> b n e', b=B)
        sep = repeat(self.sep_token, '() n e -> b n e', b=B)
        x = torch.cat([cls, x[:, :tp], sep,
                            x[:, tp:2*tp], sep,
                            x[:, 2*tp:], sep], 1)
        cor_end = 1+tp+1; sag_end = cor_end+tp+1
        x[:, :cor_end]        += self.coronal_plane
        x[:, cor_end:sag_end] += self.sagittal_plane
        x[:, sag_end:]        += self.axial_plane
        if self.training and self.token_drop_p>0:
            keep = torch.rand(B, x.size(1), device=x.device) > self.token_drop_p
            keep[:,0]=keep[:,-1]=True
            x = x * keep.unsqueeze(-1)
        return x + self.positions

class MultiHeadAttention(nn.Module):
    def __init__(self, emb=256, heads=8, p=0.1):
        super().__init__()
        self.heads, self.emb = heads, emb
        self.qkv  = nn.Linear(emb, 3*emb)
        self.proj = nn.Linear(emb, emb)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        qkv = rearrange(self.qkv(x), 'b n (h d q)->q b h n d', h=self.heads, q=3)
        q,k,v = qkv
        att = torch.softmax((q @ k.transpose(-2,-1))/(self.emb**0.5),-1)
        out = rearrange(self.drop(att) @ v, 'b h n d -> b n (h d)')
        return self.proj(out)

class ResidualAdd(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn=fn
    def forward(self,x): return x+self.fn(x)

class FeedForward(nn.Sequential):
    def __init__(self, emb, exp=2, p=0.1):
        super().__init__(nn.Linear(emb,exp*emb), nn.GELU(),
                         nn.Dropout(p), nn.Linear(exp*emb,emb))

class TransformerBlock(nn.Module):
    def __init__(self, emb=256, p=0.1, dp=0.05, exp=2):
        super().__init__()
        self.dp1, self.dp2 = DropPath(dp), DropPath(dp)
        self.res1 = ResidualAdd(nn.Sequential(nn.LayerNorm(emb),
                                              MultiHeadAttention(emb,p=p),
                                              nn.Dropout(p)))
        self.res2 = ResidualAdd(nn.Sequential(nn.LayerNorm(emb),
                                              FeedForward(emb,exp,p),
                                              nn.Dropout(p)))
    def forward(self,x):
        x = x + self.dp1(self.res1.fn(x))
        x = x + self.dp2(self.res2.fn(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth=12, emb=256, p=0.1, dp=0.05):
        super().__init__()
        dpr = np.linspace(0,dp,depth)
        self.blocks = nn.ModuleList([TransformerBlock(emb,p,dpr[i])
                                     for i in range(depth)])
    def forward(self,x):
        for blk in self.blocks: x = blk(x)
        return x

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cnn3d   = CNN3DBlock(1, cfg['out_channels'], cfg['p_cnn3d'])
        self.project = MultiPlane_MultiSlice_Extract_Project(cfg['out_channels'],
                                                             cfg['p_feat'])
        self.embed   = EmbeddingLayer(cfg['emb_size'],
                                      total_tokens=432,
                                      token_drop_p=cfg['p_token'])
        self.trans   = TransformerEncoder(cfg['depth'], cfg['emb_size'],
                                          cfg['p_transformer'], cfg['p_drop_path'])
    def forward(self, x):
        feat3d = self.cnn3d(x)
        tokens = self.project(feat3d)
        tokens = self.embed(tokens)
        tokens = self.trans(tokens)
        cls = tokens[:,0]
        return cls, feat3d

# ---------------------------------------------------------------------
# 2.  SSL heads
# ---------------------------------------------------------------------
def mlp(in_dim=256, hidden=2048, out_dim=256):
    return nn.Sequential(nn.Linear(in_dim, hidden),
                         nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Dropout(0.05),
                         nn.Linear(hidden, out_dim))

class BYOLWrapper(nn.Module):
    def __init__(self, encoder:Encoder):
        super().__init__()
        self.online_encoder  = encoder
        self.target_encoder  = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False
        self.proj_online  = mlp(256, 2048, 256)
        self.proj_target  = mlp(256, 2048, 256)
        for p in self.proj_target.parameters(): p.requires_grad = False
        self.pred         = mlp(256, 2048, 256)
        self.tau_init, self.tau_final = 0.99, 0.999
        self.tau = self.tau_init
    @torch.no_grad()
    def update_target(self, cur_epoch:int, max_epoch:int):
        self.tau = self.tau_final - (self.tau_final - self.tau_init) * \
                   (math.cos(math.pi * cur_epoch / max_epoch) + 1) / 2
        for p_o, p_t in zip(self.online_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data = p_t.data * self.tau + p_o.data * (1. - self.tau)
        for p_o, p_t in zip(self.proj_online.parameters(),
                            self.proj_target.parameters()):
            p_t.data = p_t.data * self.tau + p_o.data * (1. - self.tau)
    def forward(self, x1, x2):
        q1,_ = checkpoint(self.online_encoder, x1)
        q2,_ = checkpoint(self.online_encoder, x2)
        z1 = self.proj_online(q1); z2 = self.proj_online(q2)
        p1 = self.pred(z1);       p2 = self.pred(z2)
        with torch.no_grad():
            z1_t = self.proj_target(self.target_encoder(x1)[0])
            z2_t = self.proj_target(self.target_encoder(x2)[0])
        p1,p2,z1_t,z2_t = map(lambda t: nn.functional.normalize(t,dim=-1,p=2),
                              (p1,p2,z1_t,z2_t))
        return 2 - (p1*z2_t).sum(-1).mean() - (p2*z1_t).sum(-1).mean()

class ReconstructionDecoder(nn.Module):
    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm3d(hidden), nn.ReLU(inplace=True),
            nn.Dropout3d(0.05),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.BatchNorm3d(hidden), nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, 1)
        )
    def forward(self, feat3d): return self.dec(feat3d)

class RotationHead(nn.Module):
    def __init__(self, emb=256, classes=12): super().__init__(); self.lin = nn.Linear(emb, classes)
    def forward(self, cls): return self.lin(cls)

class JigsawHead(nn.Module):
    def __init__(self, emb=256): super().__init__(); self.lin = nn.Linear(emb, 2)
    def forward(self, cls): return self.lin(cls)

# ---------------------------------------------------------------------
# 3.  Data utilities
# ---------------------------------------------------------------------
def get_monai_augment(with_coarse: bool):
    aug = [
        RandFlipd("image", prob=0.5, spatial_axis=[0]),
        RandFlipd("image", prob=0.5, spatial_axis=[1]),
        RandFlipd("image", prob=0.5, spatial_axis=[2]),
        RandAffined("image", prob=0.7,
                    rotate_range=(np.pi/36,)*3,     # ±5°
                    translate_range=(4,4,4),
                    scale_range=(0.05,)*3,
                    padding_mode="border"),
        RandAdjustContrastd("image", prob=0.3, gamma=(0.7,1.4)),
        RandScaleIntensityd("image", factors=0.15, prob=0.3),
        RandShiftIntensityd("image", offsets=0.10, prob=0.3),
        RandGaussianNoised("image", prob=0.25, mean=0, std=0.01),
        RandGaussianSmoothd("image", prob=0.15,
                            sigma_x=(0,1), sigma_y=(0,1), sigma_z=(0,1)),
    ]
    if with_coarse:
        aug.append(RandCoarseDropoutd("image", prob=0.2,
                                      holes=4, spatial_size=(16,16,16),
                                      max_holes=4, fill_value=0))
    aug.append(EnsureTyped("image"))
    return Compose(aug)

def build_transforms(mean, std):
    load_tf = Compose([
        LoadImaged("image", image_only=False, reader="ITKReader"),
        EnsureChannelFirstd("image", strict_check=False),
        Orientationd("image", axcodes="SAR"),
        NormalizeIntensityd("image", subtrahend=mean, divisor=std),
        EnsureTyped("image")
    ])
    return load_tf, get_monai_augment(False), get_monai_augment(True)

def random_rotation_single(vol: torch.Tensor)->Tuple[torch.Tensor,int]:
    axis = random.randint(0,2); k = random.randint(0,3)
    dims_map = {0:(2,3), 1:(1,3), 2:(1,2)}
    return torch.rot90(vol,k,dims=dims_map[axis]), axis*4+k

def random_jigsaw_single(vol: torch.Tensor)->Tuple[torch.Tensor,int]:
    if random.random()<0.5:
        perm = torch.randperm(vol.shape[1]); return vol[:,perm],1
    return vol,0

def augment_batch(imgs: torch.Tensor, tf):
    return torch.stack([tf({"image": s})["image"] for s in imgs])

def create_mask(shape, ratio, device):
    return torch.rand(shape, device=device) < ratio

# ---------------------------------------------------------------------
# 4.  Training script
# ---------------------------------------------------------------------
def main():
    cfg = dict(
        resize           = (144,144,144),
        out_channels     = 64,
        emb_size         = 256,
        depth            = 12,
        p_cnn3d          = 0.05,
        p_feat           = 0.05,
        p_token          = 0.10,
        p_transformer    = 0.10,
        p_drop_path      = 0.05,
        mask_ratio       = 0.35,
        lr               = 6e-4,
        batch_size       = 12,
        epochs           = 800,
        warmup_epochs    = 10,
        lambda_recon     = 0.5,
        lambda_contrast  = 2.0,
        lambda_rot       = 0.5,
        lambda_jig       = 0.5,
        weight_decay     = 3e-6,
        save_every       = 10,
        project          = "SSL_Embryo_M3T_HYBRID"
    )
    wandb.init(project=cfg['project'], config=cfg)
    run_dir = os.path.join("ssl_runs",
                           datetime.now().strftime("%Y%m%d_%H%M%S_")+wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)

    folder = "/data/hps/home/othoma/magalab/user/othoma/Gli2_classifier/masked_cubes144_labeled_nrrd"
    files  = sorted(glob.glob(os.path.join(folder, "Scan_*_*.nrrd")))
    data_list = [dict(image=f) for f in files]
    mean,std = 1776.835584, 5603.538718
    load_tf, aug_v1, aug_v2 = build_transforms(mean, std)

    class NRRDDs(Dataset):
        def __init__(self,data,tf): super().__init__(data,tf)
    loader = DataLoader(NRRDDs(data_list, load_tf),
                        batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_cfg = {k:cfg[k] for k in ('out_channels','emb_size','depth',
                                  'p_cnn3d','p_feat','p_token',
                                  'p_transformer','p_drop_path')}
    ssl      = BYOLWrapper(Encoder(enc_cfg)).to(device)
    decoder  = ReconstructionDecoder(cfg['out_channels']).to(device)
    rot_head = RotationHead(cfg['emb_size'], 12).to(device)
    jig_head = JigsawHead(cfg['emb_size']).to(device)

    params = list(ssl.online_encoder.parameters()) + \
             list(ssl.proj_online.parameters())   + \
             list(ssl.pred.parameters())          + \
             list(decoder.parameters())           + \
             list(rot_head.parameters())          + \
             list(jig_head.parameters())
    optimiser = optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler    = GradScaler()

    def lr_schedule(epoch):
        if epoch < cfg['warmup_epochs']:
            return cfg['lr'] * (epoch+1) / cfg['warmup_epochs']
        cos_e = epoch - cfg['warmup_epochs']
        cos_T = cfg['epochs'] - cfg['warmup_epochs']
        return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * cos_e / cos_T))

    for epoch in range(cfg['epochs']):
        cur_lr = lr_schedule(epoch)
        for pg in optimiser.param_groups: pg['lr'] = cur_lr

        ssl.online_encoder.train(); decoder.train()
        rot_head.train(); jig_head.train()

        tot, c_ssl, c_rec, c_rot, c_jig, n = 0,0,0,0,0,0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for batch in pbar:
            img_cpu = batch["image"]
            img_v1 = augment_batch(img_cpu, aug_v1)
            img_v2 = augment_batch(img_cpu, aug_v2)

            img_v1 = img_v1.to(device, non_blocking=True)
            img_v2 = img_v2.to(device, non_blocking=True)

            mask = create_mask(img_v1.shape, cfg['mask_ratio'], device)
            masked = img_v1.clone(); masked[mask] = 0

            rot_imgs, rot_labels = zip(*[random_rotation_single(s) for s in img_v1])
            rot_img   = torch.stack(rot_imgs).to(device)
            rot_label = torch.tensor(rot_labels, device=device).long()

            jig_imgs, jig_labels = zip(*[random_jigsaw_single(s) for s in img_v1])
            jig_img   = torch.stack(jig_imgs).to(device)
            jig_label = torch.tensor(jig_labels, device=device).long()

            optimiser.zero_grad(set_to_none=True)
            with autocast():
                loss_ssl = ssl(masked, img_v2) * cfg['lambda_contrast']

                cls, feat3d = checkpoint(lambda y: tuple(ssl.online_encoder(y)), masked)
                recon = decoder(feat3d)
                loss_rec = nn.functional.mse_loss(recon[mask], img_v1[mask]) * cfg['lambda_recon']

                cls_rot,_ = checkpoint(lambda y: tuple(ssl.online_encoder(y)), rot_img)
                loss_rot = nn.functional.cross_entropy(rot_head(cls_rot), rot_label) * cfg['lambda_rot']

                cls_jig,_ = checkpoint(lambda y: tuple(ssl.online_encoder(y)), jig_img)
                loss_jig = nn.functional.cross_entropy(jig_head(cls_jig), jig_label) * cfg['lambda_jig']

                loss_total = loss_ssl + loss_rec + loss_rot + loss_jig

            scaler.scale(loss_total).backward()
            scaler.step(optimiser); scaler.update()
            ssl.update_target(epoch, cfg['epochs'])

            bs = img_v1.size(0)
            tot += loss_total.item()*bs; c_ssl+=loss_ssl.item()*bs; c_rec+=loss_rec.item()*bs
            c_rot+=loss_rot.item()*bs;   c_jig+=loss_jig.item()*bs; n+=bs
            pbar.set_postfix(loss=tot/n)

        wandb.log({
            "epoch": epoch+1,
            "lr": cur_lr,
            "tau": ssl.tau,
            "loss_total": tot/n,
            "loss_contrast": c_ssl/n,
            "loss_recon": c_rec/n,
            "loss_rot": c_rot/n,
            "loss_jig": c_jig/n
        })

        if (epoch+1) % cfg['save_every']==0:
            torch.save({
                "encoder": ssl.online_encoder.state_dict(),
                "proj":    ssl.proj_online.state_dict(),
                "pred":    ssl.pred.state_dict(),
                "decoder": decoder.state_dict(),
                "rot":     rot_head.state_dict(),
                "jig":     jig_head.state_dict(),
                "epoch":   epoch+1
            }, os.path.join(run_dir, f"ssl_ckpt_{epoch+1:04d}.pth"))

    torch.save(ssl.online_encoder.state_dict(),
               os.path.join(run_dir,"ssl_backbone_final.pth"))
    print(f"\n✓ Pre-training complete. Backbone saved to {run_dir}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
