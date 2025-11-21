import argparse
import os
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import FusionDataset
from models import DualUNetExtractor, DualUNetHOGClassifier
from losses import FocalLoss


def main():
    parser = argparse.ArgumentParser(description="CF-Net Stage 2: Multi-Scale Classification Fine-tuning")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--hog_path', type=str, required=True, help="Path to generated HOG features (.pth)")
    parser.add_argument('--pretrained_checkpoint', type=str, required=True, help="Stage 1 checkpoint")
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_head', type=int, default=10)
    parser.add_argument('--epochs_full', type=int, default=60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    CLASSES = ["Class1_BulkCarrier", "Class2_Containership", "Class3_Tug",
               "Class4_Fishing", "Class5_Tanker", "Class6_Dredger", "Class7_Cargo"]

    # 1. Load HOG & Prepare Data List
    print("Loading Data & HOG Features...")
    hog_dict = torch.load(args.hog_path)

    all_samples = []
    for idx, cls in enumerate(CLASSES):
        search = os.path.join(args.data_root, cls, "1bit", "*.png")
        files = glob.glob(search)
        for f in files:
            f = f.replace("\\", "/")
            rel_key = f"{cls}/1bit/{os.path.basename(f)}"

            # Only include if HOG exists (filter valid)
            if rel_key in hog_dict:
                all_samples.append({
                    "path": f,
                    "rel_key": rel_key,
                    "label": idx
                })

    # 2. Split
    labels = [s['label'] for s in all_samples]
    indices = np.arange(len(all_samples))
    train_idx, temp_idx = train_test_split(indices, train_size=0.7, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.66, stratify=[labels[i] for i in temp_idx],
                                         random_state=42)

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    # 3. Class Balancing (Upsampling to ~800 logic)
    print("Applying Class Balancing (Upsampling to ~800)...")
    train_lbls = [s['label'] for s in train_samples]
    counts = np.bincount(train_lbls, minlength=len(CLASSES))
    TARGET_N = 800

    balanced_train_samples = []
    for s in train_samples:
        c_count = counts[s['label']]
        # Calculate repetition factor
        factor = max(1, int(round(TARGET_N / c_count))) if c_count > 0 else 1
        balanced_train_samples.extend([s] * factor)

    # Shuffle
    np.random.shuffle(balanced_train_samples)
    print(f"Original Train: {len(train_samples)}, Balanced Train: {len(balanced_train_samples)}")

    # 4. Datasets & Loaders
    mean = [0.00038, 0.00038, 0.00038]
    std = [0.01725, 0.01725, 0.01725]

    train_ds = FusionDataset(balanced_train_samples, hog_dict, augment=True, mean=mean, std=std)
    val_ds = FusionDataset(val_samples, hog_dict, augment=False, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 5. Model Initialization
    print("Initializing Model...")
    extractor = DualUNetExtractor(pretrained_encoder=False)

    # Load Stage 1 Weights
    ckpt = torch.load(args.pretrained_checkpoint, map_location=device)
    st = ckpt.get('model_state_dict', ckpt.get('ema_shadow', ckpt))

    # Filter keys for encoder_b
    new_st = {}
    for k, v in st.items():
        if k.startswith('encoder_b.') or k.startswith('bottleneck_b.'):
            new_st[k] = v
    extractor.load_state_dict(new_st, strict=False)

    model = DualUNetHOGClassifier(extractor, num_classes=len(CLASSES), hog_feature_dim=128).to(device)
    criterion = FocalLoss(gamma=2.0).to(device)

    # 6. Training Phase 1: Head Only
    print("--- Phase 1: Training Classification Head Only ---")
    # Freeze Encoder
    for p in model.feature_extractor.parameters(): p.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    for epoch in range(args.epochs_head):
        model.train()
        for img, hog_v, lbl in tqdm(train_loader, desc=f"Head Ep {epoch + 1}"):
            img, hog_v, lbl = img.to(device), hog_v.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(img, hog_v)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

    # 7. Training Phase 2: Full Fine-tuning
    print("--- Phase 2: Full Fine-tuning ---")
    for p in model.parameters(): p.requires_grad = True

    # Differential Learning Rates
    params = [
        {'params': model.fc_head.parameters(), 'lr': 5e-5},
        {'params': model.fusion_block.parameters(), 'lr': 5e-5},
        {'params': model.hog_branch.parameters(), 'lr': 5e-6},
        {'params': model.feature_extractor.parameters(), 'lr': 1e-6}  # Lower LR for backbone
    ]
    optimizer = optim.AdamW(params, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_full)

    best_acc = 0.0

    for epoch in range(args.epochs_full):
        model.train()
        total_loss = 0
        for img, hog_v, lbl in tqdm(train_loader, desc=f"Full Ep {epoch + 1}"):
            img, hog_v, lbl = img.to(device), hog_v.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(img, hog_v)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, hog_v, lbl in val_loader:
                img, hog_v, lbl = img.to(device), hog_v.to(device), lbl.to(device)
                out = model(img, hog_v)
                preds = torch.argmax(out, dim=1)
                correct += (preds == lbl).sum().item()
                total += lbl.size(0)

        acc = correct / total
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f} Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_classifier.pth"))


if __name__ == "__main__":
    main()