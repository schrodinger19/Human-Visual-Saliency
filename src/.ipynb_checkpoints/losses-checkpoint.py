# src/losses.py
import torch
import torch.nn.functional as F

def kl_divergence_loss(logits, target, eps=1e-8):
    """
    Kullback–Leibler divergence between predicted saliency distribution and target.
    Automatically resizes target to match model output size.
    """
    b, c, h, w = logits.shape
    # Resize target if needed
    if target.shape[2:] != (h, w):
        target = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)

    pred = F.softmax(logits.view(b, -1), dim=1).view(b, 1, h, w)
    tgt = target.clamp(min=0.0)
    tgt = tgt / (tgt.sum(dim=(2,3), keepdim=True) + eps)
    kl = (tgt * (torch.log((tgt + eps) / (pred + eps)))).sum(dim=(2,3)).mean()
    return kl

def mse_loss(logits, target):
    """
    Mean squared error between sigmoid predictions and target.
    Automatically resizes target to match output.
    """
    b, c, h, w = logits.shape
    if target.shape[2:] != (h, w):
        target = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)
    pred = torch.sigmoid(logits)
    return F.mse_loss(pred, target)
