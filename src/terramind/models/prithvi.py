# The encoder below is adapted from prithvi_mae.py in
# https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL
# Copyright (c) IBM Corp. 2024. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Changes: encoder only (the MAE
# decoder and masking are removed), single time step, no time/location inputs,
# and a segmentation decoder added below.
"""Prithvi-EO-2.0 (100M, TL) encoder with a segmentation decoder, for E2.

Input is the 13-band Sentinel-2 chip as read by `processing.raster` (reflectance
/ 10000, clipped to [0, 1]). The wrapper picks the six bands Prithvi was trained
on (B2, B3, B4, B8A, B11, B12 ~ HLS blue, green, red, narrow NIR, SWIR1, SWIR2),
scales them back to reflectance x 10000 and normalises with the pretraining
statistics. Pretraining used 30 m HLS; Sen1Floods11 is 10 m, a domain gap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch import nn

from terramind.config import data_dir

S2_TO_HLS = [1, 2, 3, 8, 11, 12]  # indices into the 13-band Sen1Floods11 S2 stack
MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
WEIGHTS = "Prithvi_EO_V2_100M_TL.pt"


def weights_path() -> Path:
    return data_dir() / "external" / "prithvi" / WEIGHTS


def _sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=float) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_3d(dim: int, t: int, h: int, w: int) -> torch.Tensor:
    """3D sin/cos position embedding with a leading zero row for the cls token."""
    wd, hd, td = dim // 16 * 6, dim // 16 * 6, dim // 16 * 4
    we = np.tile(_sincos_1d(wd, np.arange(w)), (t * h, 1))
    he = np.tile(np.repeat(_sincos_1d(hd, np.arange(h)), w, axis=0), (t, 1))
    te = np.repeat(_sincos_1d(td, np.arange(t)), h * w, axis=0)
    emb = np.concatenate([np.zeros((1, dim)), np.concatenate([we, he, te], axis=1)], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


class PrithviEncoder(nn.Module):
    """ViT encoder with a 3D (1 x 16 x 16) patch embedding, as in Prithvi-EO-2.0."""

    def __init__(self, in_chans=6, embed_dim=768, depth=12, num_heads=12, patch=16):
        super().__init__()
        self.patch, self.embed_dim = patch, embed_dim
        # Named to match the checkpoint: encoder.patch_embed.proj, encoder.blocks, ...
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(in_chans, embed_dim, (1, patch, patch), (1, patch, patch))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, 4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, taps=(3, 6, 9, 12)) -> list[torch.Tensor]:
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        x = self.patch_embed.proj(x.unsqueeze(2)).flatten(2).transpose(1, 2)  # B, L, D
        pos = sincos_3d(self.embed_dim, 1, gh, gw).to(x.device, x.dtype)
        x = x + pos[:, 1:]
        x = torch.cat([(self.cls_token + pos[:, :1]).expand(b, -1, -1), x], dim=1)
        feats = []
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            if i in taps:
                y = self.norm(x) if i == len(self.blocks) else x
                feats.append(y[:, 1:].transpose(1, 2).reshape(b, self.embed_dim, gh, gw))
        return feats


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class PrithviSeg(nn.Module):
    """Prithvi encoder + a small UNETR-style decoder.

    Four encoder layers are brought to 1/2, 1/4, 1/8 and 1/16 of the input size
    and fused top-down, then upsampled to full resolution.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = PrithviEncoder()
        if pretrained:
            self.load_pretrained()
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        d = self.encoder.embed_dim

        def up(n):  # n transposed convolutions, each doubling the size
            layers = []
            for _ in range(n):
                layers += [
                    nn.ConvTranspose2d(d if not layers else 256, 256, 2, stride=2),
                    nn.ReLU(inplace=True),
                ]
            return nn.Sequential(*layers) if layers else nn.Conv2d(d, 256, 1)

        self.lateral = nn.ModuleList([up(3), up(2), up(1), up(0)])  # taps 3, 6, 9, 12
        self.fuse = nn.ModuleList([ConvBlock(512, 256) for _ in range(3)])
        self.head = nn.Sequential(ConvBlock(256, 64), nn.Conv2d(64, num_classes, 1))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 6, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 6, 1, 1))

    def load_pretrained(self) -> None:
        path = weights_path()
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; download Prithvi-EO-2.0-100M-TL weights")
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        enc = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
        enc = {
            k: v for k, v in enc.items() if not k.startswith(("pos_embed", "temporal", "location"))
        }
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}"
            )

    def param_groups(self, lr: float) -> list[dict]:
        """Smaller learning rate for the pretrained encoder than for the new decoder."""
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        dec = [p for n, p in self.named_parameters() if not n.startswith("encoder.")]
        groups = [{"params": dec, "lr": lr}]
        if enc:
            groups.append({"params": enc, "lr": lr * 0.05})
        return groups

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        h, w = s2.shape[-2:]
        x = (s2[:, S2_TO_HLS] * 10000.0 - self.mean) / self.std
        ph, pw = (-h) % 16, (-w) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        maps = [lat(f) for lat, f in zip(self.lateral, feats)]  # sizes 1/2, 1/4, 1/8, 1/16
        y = maps[3]
        for i, skip in zip((2, 1, 0), self.fuse):
            y = F.interpolate(y, size=maps[i].shape[-2:], mode="bilinear", align_corners=False)
            y = skip(torch.cat([y, maps[i]], dim=1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(y)[..., :h, :w]


class BandSelect(nn.Module):
    """A U-Net on the same six bands as Prithvi, to separate pretraining from band choice."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        return self.model(s2[:, S2_TO_HLS])
