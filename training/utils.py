"""Training utilities."""
import math


def cosine_warmup_schedule(epoch, cfg):
    if epoch < cfg['warmup_epochs']:
        return cfg['lr'] * (epoch+1) / cfg['warmup_epochs']
    cos_e = epoch - cfg['warmup_epochs']
    cos_T = cfg['epochs'] - cfg['warmup_epochs']
    return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * cos_e / cos_T))
