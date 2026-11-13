"""
Architectures.

- DnCNN: 17-layer residual CNN (Zhang et al., 2017), single-channel.
- UNet:  depth-2, 96-channel U-Net used by Noise2Void / Blind2Unblind, matching
         the 3.5.4 configuration (BatchNorm between conv layers).

Restormer is NOT reimplemented here: 3.5.3 initializes from the official
released checkpoint, so train_restormer.py uses the official architecture (see
that file / README) to keep checkpoint keys compatible.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DnCNN (residual; predicts the noise)
# ---------------------------------------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, channels: int = 1, num_layers: int = 17, features: int = 64):
        super().__init__()
        layers = [nn.Conv2d(channels, features, 3, padding=1, bias=True),
                  nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, channels, 3, padding=1, bias=False)]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Network estimates the residual noise; denoised = input - residual.
        return x - self.net(x)


# ---------------------------------------------------------------------------
# U-Net (depth 2, 96 features) for N2V / B2U
# ---------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Depth-2 U-Net with `feats` channels at the top level."""

    def __init__(self, channels: int = 1, depth: int = 2, feats: int = 96):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = channels
        f = feats
        chans = []
        for _ in range(depth):
            self.downs.append(_DoubleConv(c, f))
            self.pools.append(nn.MaxPool2d(2))
            chans.append(f)
            c, f = f, f * 2
        self.bottleneck = _DoubleConv(c, f)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for skip_c in reversed(chans):
            self.ups.append(nn.ConvTranspose2d(f, skip_c, 2, stride=2))
            self.up_convs.append(_DoubleConv(skip_c * 2, skip_c))
            f = skip_c
        self.head = nn.Conv2d(feats, channels, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, skip], dim=1))
        return self.head(x)


def build(method: str) -> nn.Module:
    if method == "dncnn":
        return DnCNN()
    if method in ("noise2void", "blind2unblind"):
        return UNet()
    raise ValueError(f"build() does not handle '{method}' (restormer uses its own arch).")
