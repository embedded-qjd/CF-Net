import argparse
import os
import glob
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.feature import hog
from sklearn.decomposition import PCA
from tqdm import tqdm
from models import DualUNet


def main():
    parser = argparse.ArgumentParser(description="Generate HOG features from Reconstructed Images")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to Stage 1 best_model.pth")
    parser.add_argument('--output_path', type=str, default='./hog_features.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classes (Ensure consistency with folder names)
    CLASSES = ["Class1_BulkCarrier", "Class2_Containership", "Class3_Tug",
               "Class4_Fishing", "Class5_Tanker", "Class6_Dredger", "Class7_Cargo"]

    # 1. Load Model
    print("Loading Pre-trained Model...")
    model = DualUNet(in_channels=3, out_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Handle EMA or standard state dict
    state_dict = ckpt.get('model_state_dict', ckpt.get('ema_shadow', ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. Transform
    mean = [0.00038, 0.00038, 0.00038]
    std = [0.01725, 0.01725, 0.01725]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 3. Collect Paths
    print("Collecting 1-bit images...")
    img_paths = []
    # Store relative paths to map back during classification training
    rel_map = {}

    for cls in CLASSES:
        p = os.path.join(args.data_root, cls, "1bit", "*.png")
        files = glob.glob(p)
        for f in files:
            f = f.replace("\\", "/")
            # Create a relative key for portability
            rel_key = f"{cls}/1bit/{os.path.basename(f)}"
            rel_map[rel_key] = f
            img_paths.append(rel_key)

    # 4. Inference & HOG
    print(f"Processing {len(img_paths)} images...")
    raw_hogs = []
    valid_keys = []

    with torch.no_grad():
        for key in tqdm(img_paths):
            full_path = rel_map[key]
            try:
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue

                # Preprocess
                _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                res_img = cv2.resize(bin_img, (224, 224), interpolation=cv2.INTER_NEAREST)
                t = preprocess(res_img).unsqueeze(0).to(device)

                # Reconstruct (Student Branch)
                rec = model.forward_reconstruct_b(t)

                # Convert to 8-bit for HOG
                rec_np = rec.squeeze().cpu().numpy()
                rec_uint8 = (rec_np * 255).astype(np.uint8)

                # Calculate HOG
                hog_img = cv2.resize(rec_uint8, (128, 128))
                feat = hog(hog_img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

                raw_hogs.append(feat)
                valid_keys.append(key)
            except Exception as e:
                print(f"Error {key}: {e}")

    # 5. PCA
    print("Applying PCA...")
    hog_mat = np.array(raw_hogs, dtype=np.float32)
    pca = PCA(n_components=128)
    pca_feats = pca.fit_transform(hog_mat)

    # 6. Save Dictionary
    final_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in zip(valid_keys, pca_feats)}
    torch.save(final_dict, args.output_path)
    print(f"Saved HOG features to {args.output_path}")


if __name__ == "__main__":
    main()