import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=8, embed_dim=512, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, (h, w)


class PairRelationTransformer(nn.Module):
    def __init__(
        self,
        in_channels=8,
        image_size=224,
        patch_size=16,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        num_relation_classes=3,
        num_depth_classes=3,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch = PatchEmbedding(in_channels, embed_dim, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.relation_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_relation_classes),
        )
        self.depth_order_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_depth_classes),
        )
        self.ratio_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1),
        )
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        tokens, hw = self.patch(x)
        b = tokens.shape[0]
        cls = self.cls.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        patch_tokens = tokens[:, 1:].transpose(1, 2).reshape(b, -1, hw[0], hw[1])

        heat = self.heatmap_head(patch_tokens)
        heat = F.interpolate(heat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        return {
            "relation_logits": self.relation_head(cls_out),
            "depth_order_logits": self.depth_order_head(cls_out),
            "ratio": self.ratio_head(cls_out),
            "contact_heatmap_logits": heat,
        }


def build_model(name="pair_transformer_base", **kwargs):
    if name == "pair_transformer_base":
        return PairRelationTransformer(embed_dim=512, depth=8, num_heads=8, **kwargs)
    if name == "pair_transformer_large":
        return PairRelationTransformer(embed_dim=768, depth=12, num_heads=12, **kwargs)
    if name == "pair_transformer_small":
        return PairRelationTransformer(embed_dim=384, depth=6, num_heads=6, **kwargs)
    raise ValueError(f"Unknown model: {name}")
