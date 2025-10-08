# src/dataset.py
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class SaliencyDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False):
        self.root = Path(root_dir)
        self.split = split
        with open(self.root/'index.json','r') as f:
            idx = json.load(f)
        self.ids = idx[split]
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _augment(self, img, sal):
        if np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            sal = np.ascontiguousarray(sal[:, ::-1])
        return img, sal

    def __getitem__(self, i):
        fid = self.ids[i]
        img = cv2.imread(str(self.root/self.split/'images'/f'{fid}.jpg'), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sal = cv2.imread(str(self.root/self.split/'maps'/f'{fid}.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if self.augment and self.split == 'train':
            img, sal = self._augment(img, sal)
        img = torch.from_numpy(img).permute(2,0,1)  # C,H,W
        sal = torch.from_numpy(sal)[None, ...]      # 1,H,W
        return {'image': img, 'saliency': sal, 'id': fid}