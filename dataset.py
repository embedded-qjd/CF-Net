import os
import re
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


# =============================================================================
# Dataset for Stage 1: Pre-training (Paired Images)
# =============================================================================
class SARPairedDataset(Dataset):
    def __init__(self, data_root, classes, mode='train', split_ratios=(0.8, 0.1, 0.1),
                 crop_size=224, augment=True, random_state=42):
        self.mode = mode
        self.crop_size = crop_size
        self.classes = classes
        self.augment = augment
        self.data_root = data_root

        self.samples = []
        self._load_data(split_ratios, random_state)

    def _load_data(self, split_ratios, random_state):
        print(f"Loading data from {self.data_root} ...")
        all_pairs = []
        bit1_pat = re.compile(r'_(\d+)\.')
        bit16_pat = re.compile(r'N(\d+)\.')

        for cls_idx, cls_name in enumerate(self.classes):
            dir_1bit = os.path.join(self.data_root, cls_name, "1bit")
            dir_16bit = os.path.join(self.data_root, cls_name, "16bit")

            if not os.path.isdir(dir_1bit) or not os.path.isdir(dir_16bit):
                continue

            # Map 16-bit files by index
            map_16bit = {}
            for f in os.listdir(dir_16bit):
                m = bit16_pat.search(f)
                if m: map_16bit[int(m.group(1))] = os.path.join(dir_16bit, f)

            # Match with 1-bit files
            for f in os.listdir(dir_1bit):
                m = bit1_pat.search(f)
                if m and int(m.group(1)) in map_16bit:
                    all_pairs.append({
                        "bit1": os.path.join(dir_1bit, f),
                        "bit16": map_16bit[int(m.group(1))],
                        "label": cls_idx
                    })

        # Split
        labels = [s['label'] for s in all_pairs]
        indices = np.arange(len(all_pairs))
        train_idx, temp_idx = train_test_split(indices, train_size=split_ratios[0], stratify=labels,
                                               random_state=random_state)

        if self.mode == 'train':
            self.samples = [all_pairs[i] for i in train_idx]
        else:
            temp_labels = [labels[i] for i in temp_idx]
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels,
                                                 random_state=random_state)
            self.samples = [all_pairs[i] for i in (val_idx if self.mode == 'val' else test_idx)]

    def _center_crop(self, img):
        h, w = img.shape[:2]
        cs = self.crop_size
        if h < cs or w < cs:
            img = cv2.resize(img, (cs, cs))
            return img
        y, x = (h - cs) // 2, (w - cs) // 2
        return img[y:y + cs, x:x + cs]

    def _augment_pair(self, img1, img16):
        # Geometric Augmentation
        if random.random() < 0.5:
            k = random.choice([0, 1, 2, 3])
            img1 = np.rot90(img1, k)
            img16 = np.rot90(img16, k)
        if random.random() < 0.5:
            img1 = np.fliplr(img1)
            img16 = np.fliplr(img16)

        # 1-bit Specific (Random Erasing)
        if random.random() < 0.4:
            h, w = img1.shape
            er_h, er_w = int(h * 0.2), int(w * 0.2)
            x0 = random.randint(0, w - er_w)
            y0 = random.randint(0, h - er_h)
            img1[y0:y0 + er_h, x0:x0 + er_w] = 0  # Erase

        # 16-bit Specific (Speckle Noise simulation)
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.1, img16.shape)
            img16 = img16 + img16 * noise

        return img1.copy(), img16.copy()

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img1 = cv2.imread(s['bit1'], cv2.IMREAD_GRAYSCALE)
            img16 = cv2.imread(s['bit16'], cv2.IMREAD_UNCHANGED)

            img1 = self._center_crop(img1)
            img16 = self._center_crop(img16)

            img1 = img1.astype(np.float32)
            img16 = img16.astype(np.float32)

            if self.mode == 'train' and self.augment:
                img1, img16 = self._augment_pair(img1, img16)

            # Normalize 1-bit [0, 1]
            img1 = (img1 > 127).astype(np.float32)

            # Normalize 16-bit [0, 1]
            mi, ma = img16.min(), img16.max()
            if ma - mi > 1e-6:
                img16 = (img16 - mi) / (ma - mi)
            else:
                img16 = np.zeros_like(img16)

            # Expand to 3 channels (for ResNet backbone)
            img1 = np.stack([img1] * 3, axis=0)
            img16 = np.stack([img16] * 3, axis=0)

            return torch.from_numpy(img1), torch.from_numpy(img16), s['label']
        except Exception as e:
            print(f"Error loading {s['bit1']}: {e}")
            return torch.zeros(3, self.crop_size, self.crop_size), torch.zeros(3, self.crop_size, self.crop_size), -1

    def __len__(self):
        return len(self.samples)


# =============================================================================
# Dataset for Stage 2: Classification (Image + HOG)
# =============================================================================
class FusionDataset(Dataset):
    def __init__(self, samples_info, hog_dict, crop_size=224, augment=True, mean=None, std=None):
        self.samples = samples_info
        self.hog_dict = hog_dict
        self.crop_size = crop_size
        self.augment = augment
        self.hog_dim = 128

        t_list = [transforms.ToPILImage()]
        if augment:
            t_list.append(transforms.RandomHorizontalFlip())
        t_list.append(transforms.ToTensor())
        t_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
        if mean and std:
            t_list.append(transforms.Normalize(mean, std))
        self.transform = transforms.Compose(t_list)

    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError("Img not found")
        img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        return img

    def __getitem__(self, idx):
        s = self.samples[idx]
        path_abs = s['path']
        label = s['label']

        # Retrieve HOG using relative key to ensure portability
        rel_key = s['rel_key']
        hog_feat = self.hog_dict.get(rel_key, torch.zeros(self.hog_dim))

        try:
            img_arr = self._load_img(path_abs)
            img_t = self.transform(img_arr)
            return img_t, hog_feat, torch.tensor(label, dtype=torch.long)
        except:
            # Return fallback if image read fails
            return torch.zeros(3, self.crop_size, self.crop_size), torch.zeros(self.hog_dim), torch.tensor(-1)

    def __len__(self):
        return len(self.samples)