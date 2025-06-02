"""Training utilities."""
import math


def cosine_warmup_schedule(epoch: int, cfg: dict) -> float:
    """Cosine schedule with linear warmup."""

    warmup = cfg.get('warmup_epochs', 0)
    if epoch < warmup:
        return cfg['lr'] * (epoch + 1) / warmup if warmup > 0 else cfg['lr']
    cos_e = epoch - warmup
    cos_T = cfg['epochs'] - warmup
    return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * cos_e / cos_T))
