CF-Net: Cross-Modality Fusion Network for 1-Bit Radar Classification

Official implementation of the paper "CF-Net: A Cross-Modality Reconstruction Network for High-Accuracy 1-Bit Target Classification".

Note: This project was formerly referred to as CMR-Net in early drafts. The internal class names may still reflect the underlying Dual-UNet architecture.

📖 Abstract

CF-Net is a two-stage deep learning framework designed to classify targets from extremely quantized 1-bit radar data without oversampling.

Stage 1 (Self-Supervised Pre-training): A Dual-Branch U-Net reconstructs high-fidelity 16-bit images from 1-bit inputs, learning robust structural features.

Stage 2 (Multi-Scale Classification): The pre-trained encoder is fine-tuned with a multi-scale fusion head, integrating HOG features extracted from the reconstructed images.

📂 Datasets

1. FUSAR-Ship Dataset

We utilize the public FUSAR-Ship dataset for SAR image classification.

Reference: Xiyue HOU, et al. "FUSAR-Ship: building a high-resolution SAR-AIS matchup dataset of Gaofen-3 for ship detection and recognition", Science China Information Sciences, 2020.

2. HAR Dataset (Millimeter-wave)

For Human Activity Recognition (HAR), we use our self-collected dataset.

Repository: https://github.com/embedded-qjd/HAR-Dataset-Project

⚙️ Environment

Install dependencies:

pip install -r requirements.txt


🚀 Usage

Data Structure

Organize your dataset as follows:

```text
/path/to/dataset/
    ├── Class1_BulkCarrier/
    │   ├── 1bit/
    │   └── 16bit/
    ├── Class2_Containership/
    │   ├── 1bit/
    │   └── 16bit/
    ...
 ```

1. Pre-training (Stage 1)

Train the reconstruction network to teach the encoder robust feature extraction.

python pretrain.py --data_root /path/to/dataset --save_dir ./checkpoints


2. Generate HOG Features

Extract HOG features from the reconstructed images (output of Stage 1) to ensure feature quality.

python generate_hog.py --data_root /path/to/dataset --checkpoint ./checkpoints/best_model.pth --output_path ./hog_features.pth


3. Classification Fine-tuning (Stage 2)

Fine-tune the model for the final classification task. 
python train_classifier.py --data_root /path/to/dataset --hog_path ./hog_features.pth --pretrained_checkpoint ./checkpoints/best_model.pth


⚖️ License


This project is released under the MIT License.

