"""Small U-Net: the randomly initialised baseline every later model must beat."""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2, base: int = 32, depth: int = 4):
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_channels
        for w in widths:
            self.downs.append(_block(c, w))
            c = w
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.up_blocks.append(_block(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, block in zip(self.ups, self.up_blocks):
            x = up(x)
            x = block(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)
