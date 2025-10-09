# src/metrics.py
"""
Metrics for saliency map evaluation: AUC-Judd, NSS, and fixation map binarization.
"""

import torch
import torch.nn.functional as F

def to_fixation_binary(saliency_map, threshold=0.8):
    """
    Convert saliency map (B,1,H,W) to binary fixation map based on relative threshold.
    """
    if not torch.is_tensor(saliency_map):
        saliency_map = torch.tensor(saliency_map)
    max_val = saliency_map.amax(dim=(2,3), keepdim=True).clamp(min=1e-6)
    fix = (saliency_map >= threshold * max_val).float()
    return fix


@torch.no_grad()
def auc_judd(pred_map, fix_map, eps=1e-8):
    """
    AUC-Judd implementation for saliency prediction evaluation.
    """
    b = pred_map.size(0)
    aucs = []
    for i in range(b):
        pred = pred_map[i].flatten()
        fix = fix_map[i].flatten()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + eps)

        sorted_pred, idx = torch.sort(pred, descending=True)
        fix_sorted = fix[idx]

        tp = torch.cumsum(fix_sorted, dim=0)
        fp = torch.cumsum(1 - fix_sorted, dim=0)
        tp = tp / (tp[-1] + eps)
        fp = fp / (fp[-1] + eps)

        auc = torch.trapz(tp, fp)
        aucs.append(auc.item())
    return sum(aucs) / len(aucs)


@torch.no_grad()
def nss(pred_map, fix_map, eps=1e-8):
    """
    Normalized Scanpath Saliency (NSS) metric.
    """
    b = pred_map.size(0)
    nsss = []
    for i in range(b):
        pred = pred_map[i].flatten()
        fix = fix_map[i].flatten()
        pred_norm = (pred - pred.mean()) / (pred.std() + eps)
        val = pred_norm[fix > 0.5].mean() if fix.sum() > 0 else torch.tensor(0.0, device=pred.device)
        nsss.append(val.item())
    return sum(nsss) / len(nsss)
