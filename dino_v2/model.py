import torch
import torch.nn as nn


class DINOv2Extractor(nn.Module):
    """
    DINOv2 ViT-B/14 backbone with a projection head and classification layer.
    dense_units is passed per-modality so MRI and PET can have different head sizes.
    """
    def __init__(self,
                 model_name  = "dinov2_vitb14",
                 feat_dim    = 768,
                 dense_units = 256,
                 dropout     = 0.3,
                 num_classes = 2):
        super().__init__()
        print(f"  Loading DINOv2 backbone: {model_name} (feat_dim={feat_dim})")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
            verbose=False,
        )
        self.proj   = nn.Linear(feat_dim, dense_units)
        self.relu   = nn.ReLU(inplace=True)
        self.drop   = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dense_units, num_classes)

    def forward_features(self, x):
        out     = self.backbone.forward_features(x)
        cls_tok = out["x_norm_clstoken"]
        emb     = self.relu(self.proj(cls_tok))
        emb     = self.drop(emb)
        return emb

    def forward(self, x):
        return self.fc_out(self.forward_features(x))


class MultimodalFusionModel(nn.Module):
    """
    Concatenates MRI and PET DINOv2 embeddings and passes through a fusion MLP.
    Fusion head uses CFG_FUSION dense_units/dropout independently of backbone heads.
    """
    def __init__(self, mri_ext, pet_ext,
                 fusion_dense_units = 128,
                 fusion_dropout     = 0.393,
                 num_classes        = 2):
        super().__init__()
        self.mri_stream = mri_ext
        self.pet_stream = pet_ext

        mri_out_dim = mri_ext.fc_out.in_features
        pet_out_dim = pet_ext.fc_out.in_features
        fusion_in   = mri_out_dim + pet_out_dim

        self.fusion = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_in, fusion_dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_dense_units, fusion_dense_units),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dense_units, num_classes),
        )

    def forward(self, mri, pet):
        f = torch.cat([
            self.mri_stream.forward_features(mri),
            self.pet_stream.forward_features(pet),
        ], dim=1)
        return self.fusion(f)
