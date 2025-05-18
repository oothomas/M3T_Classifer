"""Self-supervised learning heads used during pretraining."""
import math
import copy
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from .encoder import Encoder


def mlp(in_dim=256, hidden=2048, out_dim=256):
    return nn.Sequential(nn.Linear(in_dim, hidden),
                         nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Dropout(0.05),
                         nn.Linear(hidden, out_dim))

class BYOLWrapper(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.proj_online = mlp()
        self.proj_target = mlp()
        for p in self.proj_target.parameters():
            p.requires_grad = False
        self.pred = mlp()
        self.tau_init, self.tau_final = 0.99, 0.999
        self.tau = self.tau_init

    @torch.no_grad()
    def update_target(self, cur_epoch: int, max_epoch: int):
        self.tau = self.tau_final - (self.tau_final - self.tau_init) * \
                   (math.cos(math.pi * cur_epoch / max_epoch) + 1) / 2
        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data = p_t.data * self.tau + p_o.data * (1. - self.tau)
        for p_o, p_t in zip(self.proj_online.parameters(), self.proj_target.parameters()):
            p_t.data = p_t.data * self.tau + p_o.data * (1. - self.tau)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
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
    def forward(self, feat3d: Tensor) -> Tensor:
        return self.dec(feat3d)

class RotationHead(nn.Module):
    def __init__(self, emb=256, classes=12):
        super().__init__(); self.lin = nn.Linear(emb, classes)
    def forward(self, cls: Tensor) -> Tensor:
        return self.lin(cls)

class JigsawHead(nn.Module):
    def __init__(self, emb=256):
        super().__init__(); self.lin = nn.Linear(emb, 2)
    def forward(self, cls: Tensor) -> Tensor:
        return self.lin(cls)
