import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution: depthwise then pointwise (1×1)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=bias,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def _make_sepconv_vgg19_features():
    """VGG19 feature block with SeparableConv2d instead of Conv2d (random init)."""
    cfg = [64, 64, 'M',
           128, 128, 'M',
           256, 256, 256, 256, 'M',
           512, 512, 512, 512, 'M',
           512, 512, 512, 512, 'M']
    layers = []
    in_ch  = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(SeparableConv2d(in_ch, v, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_ch = v
    return nn.Sequential(*layers)


class VGG19Extractor(nn.Module):
    """VGG19-style extractor with depthwise-separable convolutions and BatchNorm."""
    def __init__(self, dense_units=128, dropout=0.5, num_classes=2):
        super().__init__()
        self.features = _make_sepconv_vgg19_features()
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
