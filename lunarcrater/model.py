"""A compact U-Net that predicts a per-pixel probability of "crater rim".

The architecture follows Ronneberger et al. (2015) and the lunar adaptation in DeepMoon
(Silburt et al., 2019), but it is deliberately small (about 1.9 M parameters with
``base=16``) so it trains on a laptop CPU in minutes.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CraterUNet(nn.Module):
    def __init__(self, base: int = 16, depth: int = 4):
        super().__init__()
        self.base = base
        self.depth = depth
        chans = [base * 2**i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        cin = 1
        for c in chans[:-1]:
            self.downs.append(ConvBlock(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chans[-2], chans[-1], dropout=0.1)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for c in reversed(chans[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.up_convs.append(ConvBlock(c * 2, c))
        self.head = nn.Conv2d(chans[0], 1, 1)

    @property
    def stride(self) -> int:
        return 2**self.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns logits
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, skip], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: CraterUNet, path, extra: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "base": model.base, "depth": model.depth, "extra": extra or {}}, path)


def load_model(path, device: str = "cpu") -> tuple[CraterUNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CraterUNet(base=ckpt.get("base", 16), depth=ckpt.get("depth", 4))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt.get("extra", {})
