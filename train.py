# train.py
import argparse, os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.dataset import SaliencyDataset
from src.model import SaliencyNet
from src.losses import kl_divergence_loss
from src.metrics import to_fixation_binary, auc_judd, nss
from src.utils import set_seed, save_checkpoint

def train_epoch(model, loader, optim, device, scaler=None):
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc='train', leave=False):
        img = batch['image'].to(device)
        sal = batch['saliency'].to(device)
        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', dtype=torch.float16):
                logits = model(img)
                loss = kl_divergence_loss(logits, sal)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(img)
            loss = kl_divergence_loss(logits, sal)
            loss.backward()
            optim.step()
        running += loss.item() * img.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses, aucs, nsss = [], [], []
    for batch in tqdm(loader, desc='val', leave=False):
        img = batch['image'].to(device)
        sal = batch['saliency'].to(device)
        logits = model(img)
        loss = kl_divergence_loss(logits, sal).item()
        pred = torch.softmax(logits.view(logits.size(0), -1), dim=1).view_as(logits)
        # Try to load fixation maps if present; else derive from GT saliency
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
    ap.add_argument('--data_root', required=True)     # e.g., data/salicon_prepared
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--out_dir', default='runs/salicon')
    ap.add_argument('--no_amp', action='store_true')
    args = ap.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = SaliencyDataset(args.data_root, 'train', augment=True)
    val_ds = SaliencyDataset(args.data_root, 'val', augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SaliencyNet(pretrained=True).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda' and not args.no_amp))

    best_nss = -1e9
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, optim, device, scaler)
        val_loss, val_auc, val_nss = validate(model, val_loader, device)
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} AUC-Judd={val_auc:.3f} NSS={val_nss:.3f}')
        if val_nss > best_nss:
            best_nss = val_nss
            save_checkpoint(os.path.join(args.out_dir, 'best.pt'), model, optim, epoch, best_nss)
    save_checkpoint(os.path.join(args.out_dir, 'last.pt'), model, optim, args.epochs, best_nss)