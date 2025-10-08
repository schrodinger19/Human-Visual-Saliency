# src/utils.py
import os, random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def save_checkpoint(path, model, optimizer, epoch, best_metric):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
        'best': best_metric
    }, path)

def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'opt' in ckpt:
        optimizer.load_state_dict(ckpt['opt'])
    return ckpt.get('epoch', 0), ckpt.get('best', None)