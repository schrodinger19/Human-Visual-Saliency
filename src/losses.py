# src/losses.py
import torch
import torch.nn.functional as F

def kl_divergence_loss(logits, target, eps=1e-8):
    # logits: (B,1,H,W) -> predicted density via softmax over spatial dims
    b, c, h, w = logits.shape
    pred = F.softmax(logits.view(b, -1), dim=1).view(b, 1, h, w)
    tgt = target.clamp(min=0.0)
    tgt = tgt / (tgt.sum(dim=(2,3), keepdim=True) + eps)
    kl = (tgt * (torch.log((tgt + eps) / (pred + eps)))).sum(dim=(2,3)).mean()
    return kl

def mse_loss(logits, target):
    pred = torch.sigmoid(logits)
    return F.mse_loss(pred, target)