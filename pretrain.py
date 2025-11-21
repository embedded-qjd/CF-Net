import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SARPairedDataset
from models import DualUNet
from losses import ReconstructionLoss


def main():
    parser = argparse.ArgumentParser(description="CF-Net Stage 1: Cross-Modality Reconstruction Pre-training")
    parser.add_argument('--data_root', type=str, required=True, help="Root path of the dataset")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Directory to save models")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Define Classes (Update folder names as needed)
    CLASSES = ["Class1_BulkCarrier", "Class2_Containership", "Class3_Tug",
               "Class4_Fishing", "Class5_Tanker", "Class6_Dredger", "Class7_Cargo"]

    print(f"--- Stage 1 Pre-training on {device} ---")

    # Data Loading
    train_ds = SARPairedDataset(args.data_root, CLASSES, mode='train', augment=True)
    val_ds = SARPairedDataset(args.data_root, CLASSES, mode='val', augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model & Optimization
    model = DualUNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    criterion = ReconstructionLoss().to(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_loss = float('inf')

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            img1, img16, labels = batch
            img1, img16, labels = img1.to(device), img16.to(device), labels.to(device)

            # Valid filter
            mask = labels != -1
            if not mask.any(): continue
            img1, img16, labels = img1[mask], img16[mask], labels[mask]

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # Target is single channel 16-bit
                target = img16[:, 0:1, :, :]

                # Forward
                out_a, out_b, feats = model(img16, img1)
                loss, loss_dict = criterion(out_a, out_b, target, feats[0], feats[1], labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Sep": f"{loss_dict['sep']:.3f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img1, img16, labels in val_loader:
                img1, img16, labels = img1.to(device), img16.to(device), labels.to(device)
                mask = labels != -1
                if not mask.any(): continue
                img1, img16, labels = img1[mask], img16[mask], labels[mask]

                target = img16[:, 0:1, :, :]
                out_a, out_b, feats = model(img16, img1)
                loss, _ = criterion(out_a, out_b, target, feats[0], feats[1], labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}")

        # Save Best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print("Model Saved!")


if __name__ == "__main__":
    main()