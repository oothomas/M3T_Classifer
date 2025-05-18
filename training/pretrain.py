"""Pretraining routine extracted from pretraining_script.py."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import wandb

from models.encoder import Encoder
from models.ssl_heads import BYOLWrapper, ReconstructionDecoder, RotationHead, JigsawHead
from data.transforms import augment_batch, create_mask, build_transforms, random_rotation_single, random_jigsaw_single
from data.dataset import NRRDDataset
from monai.data import DataLoader
from training.utils import cosine_warmup_schedule


def run_pretraining(cfg: dict) -> None:
    """Run BYOL pretraining with additional SSL objectives."""

    wandb.init(project=cfg['project'], config=cfg)
    run_dir = os.path.join("ssl_runs", wandb.run.name or wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)

    load_tf, aug_v1, aug_v2 = build_transforms(cfg['mean'], cfg['std'])
    loader = DataLoader(NRRDDataset(cfg['data_list'], load_tf),
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

    for epoch in range(cfg['epochs']):
        cur_lr = cosine_warmup_schedule(epoch, cfg)
        for pg in optimiser.param_groups: pg['lr'] = cur_lr

        ssl.online_encoder.train(); decoder.train()
        rot_head.train(); jig_head.train()

        tot = c_ssl = c_rec = c_rot = c_jig = n = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for batch in pbar:
            img_cpu = batch['image']
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
            'epoch': epoch+1,
            'lr': cur_lr,
            'tau': ssl.tau,
            'loss_total': tot/n,
            'loss_contrast': c_ssl/n,
            'loss_recon': c_rec/n,
            'loss_rot': c_rot/n,
            'loss_jig': c_jig/n
        })

        if (epoch+1) % cfg['save_every']==0:
            torch.save({
                'encoder': ssl.online_encoder.state_dict(),
                'proj':    ssl.proj_online.state_dict(),
                'pred':    ssl.pred.state_dict(),
                'decoder': decoder.state_dict(),
                'rot':     rot_head.state_dict(),
                'jig':     jig_head.state_dict(),
                'epoch':   epoch+1
            }, os.path.join(run_dir, f"ssl_ckpt_{epoch+1:04d}.pth"))

    torch.save(ssl.online_encoder.state_dict(),
               os.path.join(run_dir,"ssl_backbone_final.pth"))
    print(f"\n✓ Pre-training complete. Backbone saved to {run_dir}")
