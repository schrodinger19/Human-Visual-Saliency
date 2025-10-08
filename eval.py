# eval.py
import argparse, os
import torch
from torch.utils.data import DataLoader
from src.dataset import SaliencyDataset
from src.model import SaliencyNet
from src.metrics import auc_judd, nss, to_fixation_binary
from src.losses import kl_divergence_loss

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