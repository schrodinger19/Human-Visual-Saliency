# eval.py
<<<<<<< HEAD
import argparse, os
import torch
from torch.utils.data import DataLoader
=======
"""
Highly optimized multi-GPU evaluation script for SALICON models.

Features:
- torch.inference_mode for fastest inference path
- Mixed-precision autocast (FP16) on CUDA (Ampere+)
- Optional torch.compile for model graph optimization (PyTorch 2.x)
- DataParallel multi-GPU support (single-node)
- Fast DataLoader settings (persistent_workers, pin_memory, prefetch_factor)
- Robust checkpoint loading (handles DataParallel 'module.' prefix)
"""

import argparse
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

>>>>>>> 6b1a2b3 (Better accuracy, runs and data are excluded)
from src.dataset import SaliencyDataset
from src.model import SaliencyNet
from src.metrics import auc_judd, nss, to_fixation_binary
from src.losses import kl_divergence_loss

<<<<<<< HEAD
@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    losses, aucs, nsss = [], [], []
    for batch in loader:
        img = batch['image'].to(device)
        sal = batch['saliency'].to(device)
        logits = model(img)
        loss = kl_divergence_loss(logits, sal).item()
        pred = torch.softmax(logits.view(logits.size(0), -1), dim=1).view_as(logits)
        if 'fixation' in batch:
            fix = batch['fixation'].to(device)
        else:
            fix = to_fixation_binary(sal)
        losses.append(loss)
        aucs.append(auc_judd(pred, fix))
        nsss.append(nss(pred, fix))
    return sum(losses)/len(losses), sum(aucs)/len(aucs), sum(nsss)/len(nsss)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SaliencyDataset(args.data_root, 'val', augment=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = SaliencyNet(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])

    val_loss, val_auc, val_nss = run_eval(model, dl, device)
    print(f'val_loss={val_loss:.4f} AUC-Judd={val_auc:.3f} NSS={val_nss:.3f}')
=======
def load_checkpoint_to_model(ckpt_path, model, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    # allow ckpt either to be a dict with 'model' key or raw state_dict
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt if isinstance(ckpt, dict) else ckpt

    # handle DataParallel prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k.replace('module.', '')] = v
        state_dict = new_state

    model.load_state_dict(state_dict)
    return model

@torch.inference_mode()
def run_eval(model, loader, device, use_autocast=True):
    import torch.nn.functional as F
    model.eval()
    losses, aucs, nsss = [], [], []

    autocast_enabled = (device.type == 'cuda') and use_autocast
    autocast = torch.autocast if hasattr(torch, "autocast") else torch.cuda.amp.autocast

    with autocast(device_type='cuda', dtype=torch.float16, enabled=autocast_enabled):
        for batch in tqdm(loader, desc="Evaluating", ncols=100):
            img = batch['image'].to(device, non_blocking=True)
            sal = batch['saliency'].to(device, non_blocking=True)

            logits = model(img)
            loss = kl_divergence_loss(logits, sal).item()

            pred = torch.softmax(logits.view(logits.size(0), -1), dim=1).view_as(logits)
            ph, pw = pred.shape[2], pred.shape[3]

            if 'fixation' in batch:
                fix = batch['fixation'].to(device, non_blocking=True)
                if fix.shape[2:] != (ph, pw):
                    fix = F.interpolate(fix, size=(ph, pw), mode='nearest')
            else:
                if sal.shape[2:] != (ph, pw):
                    sal_resized = F.interpolate(sal, size=(ph, pw), mode='bilinear', align_corners=False)
                else:
                    sal_resized = sal
                fix = to_fixation_binary(sal_resized)

            losses.append(loss)
            aucs.append(auc_judd(pred, fix))
            nsss.append(nss(pred, fix))

    avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
    avg_auc = float(sum(aucs) / len(aucs)) if aucs else 0.0
    avg_nss = float(sum(nsss) / len(nsss)) if nsss else 0.0
    return avg_loss, avg_auc, avg_nss



def main():
    ap = argparse.ArgumentParser(description="Fast multi-GPU evaluation for SALICON")
    ap.add_argument('--data_root', required=True, help="Path to processed dataset (e.g. data/processed)")
    ap.add_argument('--ckpt', required=True, help="Path to model checkpoint (e.g. runs/salicon/best.pt)")
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--no_autocast', action='store_true', help="Disable autocast (FP16) even on CUDA")
    ap.add_argument('--compile', action='store_true', help="Try torch.compile(model) (PyTorch 2.x)")
    args = ap.parse_args()

    # Device and backend tuning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset + loader (tuned for high throughput)
    ds = SaliencyDataset(args.data_root, 'val', augment=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Build model
    model = SaliencyNet(pretrained=False)

    # Optionally compile the model (PyTorch 2.x)
    if args.compile:
        try:
            print(" Attempting torch.compile(model) — this may speed up inference.")
            model = torch.compile(model)
        except Exception as e:
            print(" torch.compile failed or not available:", e)

    # Load checkpoint (map to device0 for parameter loading)
    map_loc = {'cuda': device} if device.type == 'cuda' else {'cpu': device}
    # Load weights onto CPU first to be safe if checkpoint was saved on different device
    model = model.to('cpu')
    model = load_checkpoint_to_model(args.ckpt, model, map_location='cpu')

    # Multi-GPU: wrap after loaded and moved to primary device
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f" Using {torch.cuda.device_count()} GPUs for evaluation via DataParallel")
        # move model to first CUDA device
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(device)

    print(f" Running evaluation on {device} with batch_size={args.batch_size}, workers={args.num_workers}")
    # Run evaluation
    val_loss, val_auc, val_nss = run_eval(model, dl, device, use_autocast=not args.no_autocast)

    # Results
    print("\n Evaluation Complete:")
    print(f"val_loss = {val_loss:.6f}")
    print(f"AUC-Judd = {val_auc:.6f}")
    print(f"NSS       = {val_nss:.6f}")


if __name__ == '__main__':
    main()
>>>>>>> 6b1a2b3 (Better accuracy, runs and data are excluded)
