"""
Blind-spot masking utilities.

n2v_mask:  Noise2Void-style random pixel selection with neighbour-replacement
           (Krull et al., 2019). Selected pixels have their value replaced by a
           random neighbour; loss is computed only at those locations.

b2u_global_mask: Blind2Unblind global-aware cell masking (Wang et al., 2022).
           The image is partitioned into cells of size `cell` and one offset is
           masked per cell. Iterating over all `cell*cell` offsets covers every
           pixel exactly once -- used by the re-visible training scheme.
"""
from __future__ import annotations

import torch


def n2v_mask(x: torch.Tensor, rate: float, window: int = 5):
    """Return (masked_input, mask) for Noise2Void.

    x : (B, 1, H, W) noisy patch.
    mask: 1.0 at selected (blind-spot) pixels, else 0.0.
    Selected pixels are replaced by a random neighbour within `window`.
    """
    b, c, h, w = x.shape
    device = x.device
    mask = (torch.rand(b, c, h, w, device=device) < rate).float()

    half = window // 2
    dy = torch.randint(-half, half + 1, (b, c, h, w), device=device)
    dx = torch.randint(-half, half + 1, (b, c, h, w), device=device)
    yy = torch.arange(h, device=device).view(1, 1, h, 1).expand(b, c, h, w)
    xx = torch.arange(w, device=device).view(1, 1, 1, w).expand(b, c, h, w)
    sy = (yy + dy).clamp(0, h - 1)
    sx = (xx + dx).clamp(0, w - 1)
    flat = x.reshape(b, c, h * w)
    idx = (sy * w + sx).reshape(b, c, h * w)
    neigh = torch.gather(flat, 2, idx).reshape(b, c, h, w)

    masked = x * (1 - mask) + neigh * mask
    return masked, mask


def b2u_offsets(cell: int):
    """All (dy, dx) offsets within a cell, ordered for full coverage."""
    return [(i // cell, i % cell) for i in range(cell * cell)]


def b2u_cell_mask(x: torch.Tensor, cell: int, offset):
    """Boolean mask selecting one pixel per `cell`x`cell` block at `offset`.

    Returns mask (B,1,H,W) with 1.0 at masked positions.
    """
    b, c, h, w = x.shape
    dy, dx = offset
    mask = torch.zeros(b, c, h, w, device=x.device)
    mask[:, :, dy::cell, dx::cell] = 1.0
    return mask


def replace_with_neighbour(x: torch.Tensor, mask: torch.Tensor, window: int = 5):
    """Replace masked pixels with a random neighbour (used for the masked input)."""
    b, c, h, w = x.shape
    half = window // 2
    dy = torch.randint(-half, half + 1, (b, c, h, w), device=x.device)
    dx = torch.randint(-half, half + 1, (b, c, h, w), device=x.device)
    yy = torch.arange(h, device=x.device).view(1, 1, h, 1).expand(b, c, h, w)
    xx = torch.arange(w, device=x.device).view(1, 1, 1, w).expand(b, c, h, w)
    sy = (yy + dy).clamp(0, h - 1)
    sx = (xx + dx).clamp(0, w - 1)
    idx = (sy * w + sx).reshape(b, c, h * w)
    neigh = torch.gather(x.reshape(b, c, h * w), 2, idx).reshape(b, c, h, w)
    return x * (1 - mask) + neigh * mask
