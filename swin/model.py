import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SwinExtractor(nn.Module):
    """
    Swin Transformer V2-Base backbone with a 2-layer classification head.

    Architecture:
        SwinV2-B (ImageNet-1K, 256×256)
            ↓  [B, 1024]  via adaptive avg pool over patch tokens
        Dropout → Linear(1024 → dense_units) → ReLU   [forward_features]
        Linear(dense_units → num_classes)              [forward]
    """
    def __init__(self,
                 feat_dim    = 1024,
                 dense_units = 256,
                 dropout     = 0.4,
                 num_classes = 2):
        super().__init__()
        base = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)

        self.features = base.features
        self.norm     = base.norm
        self.permute  = base.permute
        self.avgpool  = base.avgpool

        self.dropout  = nn.Dropout(dropout)
        self.fc1      = nn.Linear(feat_dim, dense_units)
        self.fc2      = nn.Linear(dense_units, num_classes)

    def forward_features(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        return self.fc2(self.forward_features(x))


class MultimodalFusionModel(nn.Module):
    """Concatenates MRI and PET Swin embeddings through a fusion MLP."""
    def __init__(self, mri_ext, pet_ext,
                 dense_units=256, dropout=0.4, num_classes=2):
        super().__init__()
        self.mri_stream = mri_ext
        self.pet_stream = pet_ext
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dense_units * 2, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units),
            nn.ReLU(inplace=True),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, mri, pet):
        f = torch.cat([
            self.mri_stream.forward_features(mri),
            self.pet_stream.forward_features(pet),
        ], dim=1)
        return self.fusion(f)
