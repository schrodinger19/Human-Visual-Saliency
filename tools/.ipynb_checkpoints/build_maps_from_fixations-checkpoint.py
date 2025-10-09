# tools/build_maps_from_fixations.py
"""
Build saliency maps from SALICON fixation annotations using GPU acceleration (CuPy).

Requirements:
    pip install cupy-cuda12x opencv-python tqdm

Usage:
    python tools/build_maps_from_fixations.py \
        --ann data/raw/annotations/fixations_train2014.json \
        --images data/raw/train/train/ \
        --out data/raw/train/maps/ \
        --sigma 15
"""

import json
from pathlib import Path
import cv2
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def build_fixation_map(img_shape, fixations, sigma=15):
    """
    Create a fixation map on GPU.

    img_shape: (H, W)
    fixations: list of [row, col]  (1-indexed per SALICON → convert to 0-indexed)
    Returns float32 array on host (NumPy) normalized to 0–1.
    """
    H, W = img_shape
    m = cp.zeros((H, W), dtype=cp.float32)
    for f in fixations:
        # Convert 1-indexed → 0-indexed
        r = int(round(f[0])) - 1
        c = int(round(f[1])) - 1
        if 0 <= r < H and 0 <= c < W:
            m[r, c] += 1.0
    if m.sum() == 0:
        return cp.asnumpy(m)

    # Gaussian blur on GPU
    m = gaussian_filter(m, sigma=sigma)
    m = m / (m.max() + 1e-8)
    # Move result back to CPU for saving
    return cp.asnumpy(m)

def main(annotation_json, images_root, out_maps_root, sigma=15):
    annotation_json = Path(annotation_json)
    images_root = Path(images_root)
    out_maps_root = ensure_dir(out_maps_root)

    print(f"Loading annotation json: {annotation_json}")
    with open(annotation_json, 'r') as f:
        ann = json.load(f)

    # Build mapping: image_id → filename
    id2file = {img['id']: img['file_name'] for img in ann['images']}

    # Group fixations by image id
    from collections import defaultdict
    fix_by_img = defaultdict(list)
    for a in ann['annotations']:
        fix_by_img[a['image_id']].extend(a.get('fixations', []))

    # Loop through all images
    total = len(id2file)
    n_missing = 0
    for iid, fname in tqdm(id2file.items(), total=total, desc="Building maps"):
        stem = Path(fname).stem
        img_path = images_root / fname
        if not img_path.exists():
            # try to locate by stem (some datasets drop subfolders)
            found = None
            for p in images_root.rglob(f"{stem}.*"):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    found = p
                    break
            if found:
                img_path = found
            else:
                n_missing += 1
                continue

        # Read image to get dimensions
        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]

        fixs = fix_by_img.get(iid, [])
        salmap = build_fixation_map((H, W), fixs, sigma=sigma)

        out_path = out_maps_root / f"{stem}.png"
        cv2.imwrite(str(out_path), (salmap * 255).astype('uint8'))

    print(f"Done. Processed {total - n_missing}/{total} images. Missing images: {n_missing}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True, help='Annotation JSON (SALICON)')
    ap.add_argument('--images', required=True, help='Root dir of images (COCO_train2014_*.jpg)')
    ap.add_argument('--out', required=True, help='Output folder for saliency maps (.png)')
    ap.add_argument('--sigma', type=float, default=15.0)
    args = ap.parse_args()
    main(args.ann, args.images, args.out, sigma=args.sigma)
