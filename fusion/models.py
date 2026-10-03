"""
models.py — Backbone feature extractors.

Extractor wraps each of the 6 pretrained backbones (VGG19, ViT-B/16, DINOv2,
Swin, ConvNeXtV2, MambaOut) and exposes a frozen forward() returning the
feature vector used for MLP fusion.
"""
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as tvm
import timm

from .config import MEAN, STD, DEVICE


def get_transforms(img_size, augment):
    if augment:
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def _load_st(ckpt):
    st = torch.load(ckpt, map_location="cpu")
    return st.get("model_state", st) if isinstance(st, dict) else st


class Extractor(nn.Module):
    """
    Frozen backbone feature extractor. Returns a 1-D feature vector per image.
    Supported archs: vgg19, vitb16, dino, swin, convnext, mambaout.
    """
    def __init__(self, arch, ckpt, img_size=224):
        super().__init__()
        self.arch = arch
        if arch == "vgg19":
            m = tvm.vgg19(weights=None)
            m.load_state_dict(_load_st(ckpt), strict=False)
            self._feat = m.features; self._avgpool = m.avgpool
            self._cls  = m.classifier[:4]
            self.feat_dim = 4096
        elif arch == "vitb16":
            self.m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
            st = {(k[9:] if k.startswith("backbone.") else k): v
                  for k, v in _load_st(ckpt).items() if not k.startswith("head")}
            self.m.load_state_dict(st, strict=False)
            self.feat_dim = self.m.num_features
        elif arch == "dino":
            bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
            self._dino_bb = bb
            self._dino_proj = nn.Linear(768, 256)
            self._dino_relu = nn.ReLU(inplace=True)
            st    = _load_st(ckpt)
            bb_st = {k[9:]: v for k, v in st.items() if k.startswith("backbone.")}
            self._dino_bb.load_state_dict(bb_st, strict=True)
            self._dino_proj.weight.data.copy_(st["proj.weight"])
            self._dino_proj.bias.data.copy_(st["proj.bias"])
            self.feat_dim = 256
        elif arch == "swin":
            self.m = timm.create_model("swin_base_patch4_window7_224",
                                       pretrained=False, num_classes=0, img_size=img_size)
            st = {k: v for k, v in _load_st(ckpt).items() if not k.startswith("head")}
            self.m.load_state_dict(st, strict=False)
            self.feat_dim = self.m.num_features
        elif arch == "convnext":
            self.m = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=0)
            st = {(k[9:] if k.startswith("backbone.") else k): v
                  for k, v in _load_st(ckpt).items() if not k.startswith("head")}
            self.m.load_state_dict(st, strict=False)
            self.feat_dim = self.m.num_features
        elif arch == "mambaout":
            self.m = timm.create_model("mambaout_base", pretrained=False, num_classes=0)
            st = {(k[9:] if k.startswith("backbone.") else k): v
                  for k, v in _load_st(ckpt).items() if not k.startswith("head")}
            self.m.load_state_dict(st, strict=False)
            self.feat_dim = self.m.num_features
        else:
            raise ValueError(f"Unknown arch: {arch}")
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        if self.arch == "vgg19":
            return self._cls(torch.flatten(self._avgpool(self._feat(x)), 1))
        if self.arch == "dino":
            return self._dino_relu(self._dino_proj(
                self._dino_bb.forward_features(x)["x_norm_clstoken"]))
        if self.arch == "mambaout":
            return self.m.forward_features(x).mean([1, 2])
        return self.m(x)
