import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvNeXtV2Extractor(nn.Module):
    """
    ConvNeXtV2-Tiny backbone with a 2-layer classification head.

    Architecture:
        ConvNeXtV2-Tiny (FCMAE pre-trained, fine-tuned on ImageNet-22K+1K)
            ↓  [B, feat_dim]  via global average pooling
        Dropout → Linear(feat_dim → dense_units) → ReLU   [forward_features]
        Linear(dense_units → num_classes)                   [forward]
    """
    def __init__(self, dense_units=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0)
        feat_dim      = self.backbone.num_features
        self.dropout  = nn.Dropout(dropout)
        self.fc1      = nn.Linear(feat_dim, dense_units)
        self.fc2      = nn.Linear(dense_units, num_classes)

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.forward_features(x))


class MultimodalFusionModel(nn.Module):
    """Concatenates MRI and PET ConvNeXtV2 embeddings through a fusion MLP."""
    def __init__(self, mri_ext, pet_ext,
                 dense_units=256, dropout=0.3, num_classes=2):
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
