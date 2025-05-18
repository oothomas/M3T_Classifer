"""Core encoder architecture shared across pretraining, finetuning and saliency.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from einops import rearrange, repeat

class DropPath(nn.Module):
    """Stochastic depth layer used inside the Transformer."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class CNN3DBlock(nn.Module):
    """Simple two-layer 3D CNN used at the start of the encoder."""

    def __init__(self, in_channels: int, out_channels: int, drop_p: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 5, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 5, padding=2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(drop_p)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(self.relu(self.bn2(self.conv2(x))))
        return x

class MultiPlane_MultiSlice_Extract_Project(nn.Module):
    """Extract 2D plane features and project them to token embeddings."""

    def __init__(self, out_channels: int, feat_drop_p: float = 0.05):
        super().__init__()
        self.CNN_2D = models.resnet50(weights=None)
        self.CNN_2D.conv1 = nn.Conv2d(out_channels, 64, 7, 2, 3, bias=False)
        self.CNN_2D.fc = nn.Identity()
        self.feat_drop = nn.Dropout(feat_drop_p)
        self.proj = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Linear(512, 256)
        )

    def forward(self, t3d):
        B, C, D, H, W = t3d.shape
        cor = torch.cat(torch.split(t3d, 1, dim=2), dim=2)
        sag = torch.cat(torch.split(t3d, 1, dim=3), dim=3)
        axi = torch.cat(torch.split(t3d, 1, dim=4), dim=4)
        S = torch.cat(((cor * t3d).permute(0,2,1,3,4),
                       (sag * t3d).permute(0,3,1,2,4),
                       (axi * t3d).permute(0,4,1,2,3)), dim=1).reshape(-1, C, H, W)
        f2d = self.CNN_2D(S).view(B, -1, 2048)
        return self.proj(self.feat_drop(f2d))

class EmbeddingLayer(nn.Module):
    """Add positional, plane and special token embeddings."""

    def __init__(self, emb_size: int, total_tokens: int, token_drop_p: float):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.sep_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.coronal_plane = nn.Parameter(torch.randn(1, emb_size))
        self.sagittal_plane = nn.Parameter(torch.randn(1, emb_size))
        self.axial_plane = nn.Parameter(torch.randn(1, emb_size))
        self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))
        self.token_drop_p = token_drop_p

    def forward(self, x):
        B,T,_ = x.shape; tp = T // 3
        cls = repeat(self.cls_token, '() n e -> b n e', b=B)
        sep = repeat(self.sep_token, '() n e -> b n e', b=B)
        x = torch.cat([cls, x[:,:tp], sep, x[:,tp:2*tp], sep, x[:,2*tp:], sep], 1)
        cor_end = 1+tp+1; sag_end = cor_end+tp+1
        x[:,:cor_end] += self.coronal_plane
        x[:,cor_end:sag_end] += self.sagittal_plane
        x[:,sag_end:] += self.axial_plane
        if self.training and self.token_drop_p>0:
            keep = torch.rand(B, x.size(1), device=x.device) > self.token_drop_p
            keep[:,0]=keep[:,-1]=True
            x = x * keep.unsqueeze(-1)
        return x + self.positions

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, emb=256, heads=8, p=0.1):
        super().__init__()
        self.heads, self.emb = heads, emb
        self.qkv = nn.Linear(emb, 3 * emb)
        self.proj = nn.Linear(emb, emb)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), 'b n (h d q)->q b h n d', h=self.heads, q=3)
        q,k,v = qkv
        att = torch.softmax((q @ k.transpose(-2,-1))/(self.emb**0.5),-1)
        out = rearrange(self.drop(att) @ v, 'b h n d -> b n (h d)')
        return self.proj(out)

class ResidualAdd(nn.Module):
    """Utility module to add a residual connection."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

class FeedForward(nn.Sequential):
    """Position-wise feed-forward network."""

    def __init__(self, emb, exp=2, p=0.1):
        super().__init__(
            nn.Linear(emb, exp * emb), nn.GELU(), nn.Dropout(p), nn.Linear(exp * emb, emb)
        )

class TransformerBlock(nn.Module):
    """A single Transformer encoder block with stochastic depth."""

    def __init__(self, emb=256, p=0.1, dp=0.05, exp=2):
        super().__init__()
        self.dp1, self.dp2 = DropPath(dp), DropPath(dp)
        self.res1 = ResidualAdd(
            nn.Sequential(nn.LayerNorm(emb), MultiHeadAttention(emb, p=p), nn.Dropout(p))
        )
        self.res2 = ResidualAdd(
            nn.Sequential(nn.LayerNorm(emb), FeedForward(emb, exp, p), nn.Dropout(p))
        )

    def forward(self, x):
        x = x + self.dp1(self.res1.fn(x))
        x = x + self.dp2(self.res2.fn(x))
        return x

class TransformerEncoder(nn.Module):
    """Sequence of :class:`TransformerBlock` layers."""

    def __init__(self, depth=12, emb=256, p=0.1, dp=0.05):
        super().__init__()
        dpr = np.linspace(0, dp, depth)
        self.blocks = nn.ModuleList([TransformerBlock(emb, p, dpr[i]) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class Encoder(nn.Module):
    """The main model encoding a 3D volume into a sequence representation."""

    def __init__(self, cfg):
        super().__init__()
        self.cnn3d = CNN3DBlock(1, cfg["out_channels"], cfg["p_cnn3d"])
        self.project = MultiPlane_MultiSlice_Extract_Project(
            cfg["out_channels"], cfg["p_feat"]
        )
        self.embed = EmbeddingLayer(
            cfg["emb_size"], total_tokens=432, token_drop_p=cfg["p_token"]
        )
        self.trans = TransformerEncoder(
            cfg["depth"], cfg["emb_size"], cfg["p_transformer"], cfg["p_drop_path"]
        )

    def forward(self, x):
        feat3d = self.cnn3d(x)
        tokens = self.project(feat3d)
        tokens = self.embed(tokens)
        tokens = self.trans(tokens)
        cls = tokens[:, 0]
        return cls, feat3d
