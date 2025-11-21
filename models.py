import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# =============================================================================
# Part 1: Basic Components (Encoder & Blocks)
# =============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet-34 Encoder that returns multi-scale features.
    Not using ImageNet weights during inference if specified.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet34(weights=weights)
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

    def forward(self, x):
        # Returns [s0, s1, s2, s3] for skip connections and multi-scale fusion
        s0 = self.encoder0(x)
        s1 = self.encoder1(s0)
        s2 = self.encoder2(s1)
        s3 = self.encoder3(s2)
        return [s0, s1, s2, s3]


class CrossAttention(nn.Module):
    """
    Cross-Attention module for fusing Teacher (16-bit) and Student (1-bit) features.
    """

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_a, x_b):
        B, C, H, W = x_a.size()
        q = self.query(x_a).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x_b).view(B, -1, H * W)
        v = self.value(x_b).view(B, -1, H * W).permute(0, 2, 1)
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).view(B, -1, H, W)
        return x_a + self.gamma * out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_conn):
        x = self.up(x)
        x = torch.cat([x, skip_conn], dim=1)
        return self.conv(x)


# =============================================================================
# Part 2: Stage 1 Model (DualUNet for Reconstruction)
# =============================================================================

class DualUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained_encoder=True):
        super().__init__()
        # Branch A: Teacher (16-bit), Branch B: Student (1-bit)
        self.encoder_a = ResNetEncoder(pretrained=pretrained_encoder)
        self.encoder_b = ResNetEncoder(pretrained=pretrained_encoder)

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained_encoder else None
        self.bottleneck_a = resnet34(weights=weights).layer4
        self.bottleneck_b = resnet34(weights=weights).layer4

        self.cross_attn = CrossAttention(channels=512)

        # Decoders
        self.decoder_a4 = DecoderBlock(512, 256, 256)
        self.decoder_a3 = DecoderBlock(256, 128, 128)
        self.decoder_a2 = DecoderBlock(128, 64, 64)
        self.decoder_a1 = DecoderBlock(64, 64, 64)

        self.decoder_b4 = DecoderBlock(512, 256, 256)
        self.decoder_b3 = DecoderBlock(256, 128, 128)
        self.decoder_b2 = DecoderBlock(128, 64, 64)
        self.decoder_b1 = DecoderBlock(64, 64, 64)

        self.final_up_a = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv_a = nn.Conv2d(32, out_channels, kernel_size=1)
        self.final_up_b = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv_b = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x_a, x_b):
        # Returns reconstruction output and bottleneck features
        a_s0, a_s1, a_s2, a_s3 = self.encoder_a(x_a)
        b_s0, b_s1, b_s2, b_s3 = self.encoder_b(x_b)

        a_bottle = self.bottleneck_a(a_s3)
        b_bottle = self.bottleneck_b(b_s3)

        fused = self.cross_attn(a_bottle, b_bottle)

        # Teacher Decoder
        d_a = self.decoder_a4(fused, a_s3)
        d_a = self.decoder_a3(d_a, a_s2)
        d_a = self.decoder_a2(d_a, a_s1)
        d_a = self.decoder_a1(d_a, a_s0)
        out_a = torch.sigmoid(self.final_conv_a(self.final_up_a(d_a)))

        # Student Decoder
        d_b = self.decoder_b4(fused, b_s3)
        d_b = self.decoder_b3(d_b, b_s2)
        d_b = self.decoder_b2(d_b, b_s1)
        d_b = self.decoder_b1(d_b, b_s0)
        out_b = torch.sigmoid(self.final_conv_b(self.final_up_b(d_b)))

        return out_a, out_b, [a_bottle, b_bottle]

    def forward_reconstruct_b(self, x_b):
        """Inference mode: Reconstruct from 1-bit input only (Student Branch)."""
        b_s0, b_s1, b_s2, b_s3 = self.encoder_b(x_b)
        b_bottle = self.bottleneck_b(b_s3)

        d_b = self.decoder_b4(b_bottle, b_s3)
        d_b = self.decoder_b3(d_b, b_s2)
        d_b = self.decoder_b2(d_b, b_s1)
        d_b = self.decoder_b1(d_b, b_s0)
        out_b = torch.sigmoid(self.final_conv_b(self.final_up_b(d_b)))
        return out_b


# =============================================================================
# Part 3: Stage 2 Model (Classifier with Multi-Scale Fusion)
# =============================================================================

class DualUNetExtractor(nn.Module):
    """Wrapper to extract multi-scale features from the pre-trained encoder."""

    def __init__(self, pretrained_encoder=True):
        super().__init__()
        self.encoder_b = ResNetEncoder(pretrained=pretrained_encoder)
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained_encoder else None
        self.bottleneck_b = resnet34(weights=weights).layer4

    def forward(self, x):
        encoder_features = self.encoder_b(x)  # [s0, s1, s2, s3]
        bottleneck_features = self.bottleneck_b(encoder_features[-1])
        # Return list of all scale features
        return encoder_features + [bottleneck_features]


class DualUNetHOGClassifier(nn.Module):
    """Final Classification Network: CNN Multi-Scale Features + HOG Features"""

    def __init__(self, dual_unet_extractor, num_classes, hog_feature_dim=128):
        super().__init__()
        self.feature_extractor = dual_unet_extractor

        # Scale processors for features: [64, 64, 128, 256, 512]
        cnn_feature_channels = [64, 64, 128, 256, 512]
        self.target_cnn_dim = 256

        self.scale_processors = nn.ModuleList()
        for in_channels in cnn_feature_channels:
            self.scale_processors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, self.target_cnn_dim),
                nn.ReLU()
            ))

        self.hog_branch = nn.Sequential(
            nn.Linear(hog_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, 128)
        )

        fused_dim = self.target_cnn_dim + 128  # 256 (CNN avg) + 128 (HOG)

        self.fusion_block = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc_head = nn.Linear(512, num_classes)

    def forward(self, x_img, x_hog):
        # 1. Extract Multi-scale Features
        all_features = self.feature_extractor(x_img)

        # 2. Process and Aggregate Scales
        processed_vectors = [proc(f) for proc, f in zip(self.scale_processors, all_features)]
        multiscale_cnn_vector = torch.mean(torch.stack(processed_vectors, dim=0), dim=0)

        # 3. Process HOG
        hog_vec = self.hog_branch(x_hog)

        # 4. Fusion
        fused = torch.cat((multiscale_cnn_vector, hog_vec), dim=1)
        return self.fc_head(self.fusion_block(fused))