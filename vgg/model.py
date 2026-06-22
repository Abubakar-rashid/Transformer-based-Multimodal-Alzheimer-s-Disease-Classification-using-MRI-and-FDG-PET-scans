import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG19Extractor(nn.Module):
    """Pretrained VGG19 backbone with GAP and a 2-layer classification head."""
    def __init__(self, dense_units=128, dropout=0.5, num_classes=2):
        super().__init__()
        base          = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(dropout)
        self.fc1      = nn.Linear(512, dense_units)
        self.fc2      = nn.Linear(dense_units, num_classes)

    def forward_features(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        return self.fc2(self.forward_features(x))


class MultimodalFusionModel(nn.Module):
    def __init__(self, mri_ext, pet_ext,
                 dense_units=128, dropout=0.5, num_classes=2):
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
