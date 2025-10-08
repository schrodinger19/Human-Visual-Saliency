# scripts/prepare_salicon.py
import argparse, os, zipfile, shutil, json, random
from pathlib import Path
import cv2
import numpy as np

def unzip_to(src_zip, dst_dir):
    with zipfile.ZipFile(src_zip, 'r') as zf:
        zf.extractall(dst_dir)

def resize_and_save(src, dst, size, is_gray=False):
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR)
    if img is None:
        return False
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if min(img.shape[:2]) >= size else cv2.INTER_LINEAR
    img = cv2.resize(img, (size, size), interpolation=interp)
    if is_gray:
        cv2.imwrite(str(dst), img)
    else:
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return True

def prepare_split(file_ids, split_ratio=0.1, seed=42):
    rng = random.Random(seed)
    ids = list(file_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * split_ratio)
    return ids[n_val:], ids[:n_val]

def build_index(out_dir):
    out_dir = Path(out_dir)
    with open(out_dir/'index.json', 'w') as f:
        json.dump({
            "train": sorted([p.stem for p in (out_dir/'train'/'images').glob('*.jpg')]),
            "val": sorted([p.stem for p in (out_dir/'val'/'images').glob('*.jpg')])
        }, f)

def prep_salicon(raw_dir, out_dir, size):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    for s in ['train','val']:
        for sub in ['images','maps','fixations']:
            src_zip = raw_dir / sub / f'{s}.zip'
            tmp_dir = raw_dir / 'tmp' / sub / s
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            unzip_to(src_zip, tmp_dir)

    # Map COCO filenames to IDs without extension
    def collect_ids(img_root):
        return [p.stem for p in Path(img_root).glob('*.jpg')]

    all_ids = collect_ids(raw_dir/'tmp'/'images'/'train')
    train_ids, val_ids = prepare_split(all_ids, split_ratio=0.1)

    for split, ids in [('train', train_ids), ('val', val_ids)]:
        (out_dir/split/'images').mkdir(parents=True, exist_ok=True)
        (out_dir/split/'maps').mkdir(parents=True, exist_ok=True)
        (out_dir/split/'fixations').mkdir(parents=True, exist_ok=True)
        for fid in ids:
            img = raw_dir/'tmp'/'images'/'train'/f'{fid}.jpg'
            if not img.exists():
                img = raw_dir/'tmp'/'images'/'val'/f'{fid}.jpg'
            sal = raw_dir/'tmp'/'maps'/'train'/f'{fid}.png'
            if not sal.exists():
                sal = raw_dir/'tmp'/'maps'/'val'/f'{fid}.png'
            fix = raw_dir/'tmp'/'fixations'/'train'/f'{fid}.mat'
            if not fix.exists():
                fix = raw_dir/'tmp'/'fixations'/'val'/f'{fid}.mat'
            # Some SALICON fixation maps are PNGs; if .mat missing, try .png
            if not fix.exists():
                fix_png = raw_dir/'tmp'/'fixations'/'train'/f'{fid}.png'
                if not fix_png.exists():
                    fix_png = raw_dir/'tmp'/'fixations'/'val'/f'{fid}.png'
                fix = fix_png if fix_png.exists() else None

            ok_img = resize_and_save(img, out_dir/split/'images'/f'{fid}.jpg', size, is_gray=False)
            ok_map = resize_and_save(sal, out_dir/split/'maps'/f'{fid}.png', size, is_gray=True)
            if fix and fix.suffix.lower() == '.png':
                ok_fix = resize_and_save(fix, out_dir/split/'fixations'/f'{fid}.png', size, is_gray=True)
            else:
                # If only .mat fixations available, we skip here; eval will handle missing fixation by using binarized saliency peaks
                ok_fix = True
            if not (ok_img and ok_map and ok_fix):
                for sub in ['images','maps','fixations']:
                    p = out_dir/split/sub/f'{fid}.jpg'
                    if p.exists(): p.unlink(missing_ok=True)
    build_index(out_dir)

def prep_mit1003(raw_dir, out_dir, size):
    # Expect raw_dir contains images.zip and maps.zip (or fixation maps)
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    for sub in ['images','maps']:
        tmp_dir = raw_dir/'tmp'/sub
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Try common names
        zcands = [raw_dir/f'{sub}.zip', raw_dir/f'{sub.upper()}.zip']
        zcands = [p for p in zcands if p.exists()]
        if zcands:
            unzip_to(zcands[0], tmp_dir)
    ids = [p.stem for p in (raw_dir/'tmp'/'images').rglob('*.jpg')]
    train_ids, val_ids = prepare_split(ids, split_ratio=0.2)
    for split, split_ids in [('train',train_ids),('val',val_ids)]:
        (out_dir/split/'images').mkdir(parents=True, exist_ok=True)
        (out_dir/split/'maps').mkdir(parents=True, exist_ok=True)
        for fid in split_ids:
            imgp = next((raw_dir/'tmp'/'images').rglob(f'{fid}.jpg'))
            mapp = next(((raw_dir/'tmp'/'maps').rglob(f'{fid}.png')), None)
            if mapp is None:
                mapp = next(((raw_dir/'tmp'/'maps').rglob(f'{fid}.jpg')), None)
            ok1 = resize_and_save(imgp, out_dir/split/'images'/f'{fid}.jpg', size, is_gray=False)
            ok2 = resize_and_save(mapp, out_dir/split/'maps'/f'{fid}.png', size, is_gray=True) if mapp else False
    build_index(out_dir)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['SALICON','MIT1003'], required=True)
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    args = ap.parse_args()
    if args.dataset == 'SALICON':
        prep_salicon(args.raw_dir, args.out_dir, args.img_size)
    else:
        prep_mit1003(args.raw_dir, args.out_dir, args.img_size)