import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GhostConv(nn.Module):
    """
    Ghost module (Han et al., GhostNet CVPR 2020).
    Primary 1×1 conv produces s=out_ch//2 intrinsic maps;
    a cheap depthwise conv generates s ghost maps. Concat → out_ch channels.
    ~50% FLOPs vs standard 3×3 Conv2d.
    """
    def __init__(self, in_ch, out_ch,
                 kernel_size=1, stride=1, padding=0,
                 dw_size=3, ratio=2, use_relu=True):
        super().__init__()
        assert out_ch % ratio == 0, "out_ch must be divisible by ratio"
        init_ch = out_ch // ratio

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True) if use_relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_ch, init_ch, dw_size, stride=1,
                      padding=dw_size // 2, groups=init_ch, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True) if use_relu else nn.Identity(),
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        ghost   = self.cheap_operation(primary)
        return torch.cat([primary, ghost], dim=1)


def _make_ghost_vgg19_features():
    """VGG19 feature block with GhostConv replacing every Conv2d+ReLU pair."""
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
            layers.append(GhostConv(in_ch, v,
                                    kernel_size=1, stride=1, padding=0,
                                    dw_size=3))
            in_ch = v
    return nn.Sequential(*layers)


class GhostVGG19Extractor(nn.Module):
    """
    VGG19 feature extractor with Ghost Convolutions.
    Primary branches are warm-started from the centre pixel of pretrained VGG19 3×3 kernels.
    """
    def __init__(self, dense_units=128, dropout=0.5, num_classes=2):
        super().__init__()
        self.features = _make_ghost_vgg19_features()
        self._init_from_pretrained()
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(512, dense_units)
        self.fc2     = nn.Linear(dense_units, num_classes)

    def _init_from_pretrained(self):
        pretrained = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        pre_convs  = [m for m in pretrained.features if isinstance(m, nn.Conv2d)]
        ghost_mods = [m for m in self.features if isinstance(m, GhostConv)]

        for ghost_mod, pre_conv in zip(ghost_mods, pre_convs):
            primary_conv = ghost_mod.primary_conv[0]
            with torch.no_grad():
                out_half = primary_conv.weight.shape[0]
                primary_conv.weight.copy_(
                    pre_conv.weight[:out_half, :, 1:2, 1:2]
                )
        print("  [GhostVGG19] Primary branches initialised from pretrained VGG19 "
              "(centre-pixel of 3×3 kernels)")

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
