# scripts/prepare_salicon.py
"""
GPU-accelerated preprocessing for SALICON and MIT1003 datasets.

Features:
- Uses CUDA (cv2.cuda + CuPy) for fast resize and color conversion.
- Supports already unzipped folder structures (no zips required).
- Falls back to CPU if GPU unavailable.
"""

import argparse, os, json, random, shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Try GPU acceleration
try:
    import cupy as cp
    USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    cp = None
    USE_CUDA = False

def resize_and_save(src, dst, size, is_gray=False):
    """Resize an image using GPU if available, else CPU."""
    src = Path(src)
    if not src.exists():
        return False

    # ---- GPU path ----
    if USE_CUDA:
        try:
            # Load image to GPU
            if is_gray:
                img_gpu = cv2.cuda.imread(str(src), cv2.IMREAD_GRAYSCALE)
            else:
                img_gpu = cv2.cuda.imread(str(src), cv2.IMREAD_COLOR)

            if img_gpu is None:
                return False

            # Resize on GPU
            img_resized_gpu = cv2.cuda.resize(img_gpu, (size, size), interpolation=cv2.INTER_AREA)

            # Convert color if not gray
            if not is_gray:
                img_resized_gpu = cv2.cuda.cvtColor(img_resized_gpu, cv2.COLOR_BGR2RGB)

            # Download to CPU memory
            img_resized = img_resized_gpu.download()

            # Write image
            if is_gray:
                cv2.imwrite(str(dst), img_resized)
            else:
                cv2.imwrite(str(dst), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            return True
        except Exception:
            # fallback to CPU path if cv2.cuda.imread not supported
            pass

    # ---- CPU fallback ----
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
    index = {
        "train": sorted([p.stem for p in (out_dir / "train" / "images").glob("*.jpg")]),
        "val": sorted([p.stem for p in (out_dir / "val" / "images").glob("*.jpg")])
    }
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)
    print(f" Built index.json with {len(index['train'])} train and {len(index['val'])} val samples.")


def prep_salicon(raw_dir, out_dir, size):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)

    # Assume unzipped folder structure:
    # data/raw/train/{images,maps,fixations}, same for val
    splits = ['train', 'val']
    for split in splits:
        print(f"\n Processing split: {split}")
        img_root = raw_dir / split / "train" if (raw_dir / split / "train").exists() else raw_dir / split / "images"
        map_root = raw_dir / split / "maps"
        fix_root = raw_dir / split / "fixations"

        # collect image IDs
        img_files = list(img_root.glob("*.jpg"))
        ids = [p.stem for p in img_files]
        print(f"Found {len(ids)} images for {split} split.")

        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "maps").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "fixations").mkdir(parents=True, exist_ok=True)

        for fid in tqdm(ids, desc=f"Resizing {split}", ncols=80):
            img = img_root / f"{fid}.jpg"
            sal = map_root / f"{fid}.png"
            fix = fix_root / f"{fid}.png"

            ok_img = resize_and_save(img, out_dir / split / "images" / f"{fid}.jpg", size, is_gray=False)
            ok_map = resize_and_save(sal, out_dir / split / "maps" / f"{fid}.png", size, is_gray=True)
            ok_fix = resize_and_save(fix, out_dir / split / "fixations" / f"{fid}.png", size, is_gray=True) if fix.exists() else True

            if not (ok_img and ok_map and ok_fix):
                for sub in ['images', 'maps', 'fixations']:
                    for ext in ['.jpg', '.png']:
                        p = out_dir / split / sub / f"{fid}{ext}"
                        if p.exists():
                            p.unlink(missing_ok=True)

    build_index(out_dir)
    print(f"\n Preprocessing complete. Output saved to: {out_dir}")


def prep_mit1003(raw_dir, out_dir, size):
    print("MIT1003 preprocessing not GPU-accelerated. Running on CPU.")
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    ids = [p.stem for p in (raw_dir / "images").rglob("*.jpg")]
    train_ids, val_ids = prepare_split(ids, split_ratio=0.2)
    for split, split_ids in [('train', train_ids), ('val', val_ids)]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "maps").mkdir(parents=True, exist_ok=True)
        for fid in tqdm(split_ids, desc=f"Resizing {split}", ncols=80):
            imgp = next((raw_dir / "images").rglob(f"{fid}.jpg"))
            mapp = next(((raw_dir / "maps").rglob(f"{fid}.png")), None)
            if mapp is None:
                mapp = next(((raw_dir / "maps").rglob(f"{fid}.jpg")), None)
            ok1 = resize_and_save(imgp, out_dir / split / "images" / f"{fid}.jpg", size, is_gray=False)
            ok2 = resize_and_save(mapp, out_dir / split / "maps" / f"{fid}.png", size, is_gray=True) if mapp else False
    build_index(out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["SALICON", "MIT1003"], required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    print(f" CUDA available: {USE_CUDA}")
    if args.dataset == "SALICON":
        prep_salicon(args.raw_dir, args.out_dir, args.img_size)
    else:
        prep_mit1003(args.raw_dir, args.out_dir, args.img_size)
