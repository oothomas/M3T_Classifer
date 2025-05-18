"""Finetuning routine extracted from finetuning_script.py."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from models.classifier import M3T_Edema
from data.transforms import build_transforms
from data.dataset import NRRDDataset
from monai.data import DataLoader
from training.utils import cosine_warmup_schedule


def run_finetuning(cfg):
    wandb.init(project=cfg['project'], config=cfg)
    run_dir = os.path.join('runs', wandb.run.name or wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)

    train_tf, val_tf = build_transforms(cfg['mean'], cfg['std'])[:2]
    train_loader = DataLoader(NRRDDataset(cfg['train_list'], train_tf),
                              batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(NRRDDataset(cfg['val_list'], val_tf),
                              batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc_cfg = {k:cfg[k] for k in ('out_channels','emb_size','depth',
                                  'p_cnn3d','p_feat','p_token',
                                  'p_transformer','p_drop_path')}
    model = M3T_Edema(enc_cfg).to(device)

    if cfg.get('ssl_ckpt') and os.path.isfile(cfg['ssl_ckpt']):
        ckpt = torch.load(cfg['ssl_ckpt'], map_location='cpu')
        msg  = model.encoder.load_state_dict(ckpt['encoder'], strict=False)
        print(f"Loaded SSL weights with {len(msg.missing_keys)} missing keys")

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler    = GradScaler()

    counts = [0,0]
    for s in cfg['train_list']:
        counts[s['label_edema']] += 1
    w = torch.tensor([sum(counts)/c for c in counts], device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    for epoch in range(cfg['epochs']):
        model.train()
        tloss = tacc = n = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{cfg['epochs']}")
        for batch in pbar:
            imgs = batch['image'].to(device)
            labs = batch['label_edema'].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits = model(imgs)
                loss   = criterion(logits, labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            acc = (logits.argmax(1)==labs).float().mean().item()
            tloss += loss.item()*labs.size(0)
            tacc  += acc*labs.size(0)
            n     += labs.size(0)
            pbar.set_postfix(loss=tloss/n, acc=tacc/n)

        model.eval()
        vloss = vacc = vn = 0
        with torch.no_grad(), autocast(dtype=torch.float16):
            for batch in tqdm(val_loader, desc='Val'):
                imgs = batch['image'].to(device)
                labs = batch['label_edema'].to(device).long()
                logits = model(imgs)
                loss   = criterion(logits, labs)
                acc    = (logits.argmax(1)==labs).float().mean().item()
                vloss += loss.item()*labs.size(0)
                vacc  += acc*labs.size(0)
                vn    += labs.size(0)

        wandb.log(dict(epoch=epoch+1,
                       train_loss=tloss/n,
                       val_loss=vloss/vn,
                       train_acc=tacc/n,
                       val_acc=vacc/vn))

        if (epoch+1) % cfg['save_every']==0:
            ckpt = dict(epoch=epoch+1,
                        model=model.state_dict(),
                        opt=optimizer.state_dict(),
                        scheduler=scheduler.state_dict())
            torch.save(ckpt, os.path.join(run_dir, f"checkpoint_epoch{epoch+1:03d}.pth"))

        scheduler.step()
    print(f"Training complete. Run dir: {run_dir}")
