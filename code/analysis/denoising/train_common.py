"""
Shared training utilities: a single loop driven by a per-step loss closure, so
all four methods reuse the same scheduling, AMP, logging and checkpointing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

# Version-robust AMP: newer torch moved these to torch.amp with a device arg;
# older torch (e.g. the CUDA 12.4 builds on Astraea) uses torch.cuda.amp.
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast

    def make_scaler(enabled):
        return _GradScaler("cuda", enabled=enabled)

    def amp_autocast(enabled):
        return _autocast("cuda", enabled=enabled)
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast

    def make_scaler(enabled):
        return _GradScaler(enabled=enabled)

    def amp_autocast(enabled):
        return _autocast(enabled=enabled)


def cosine_lr(optimizer, base_lr, epoch, total_epochs, warmup=2):
    import math
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        t = (epoch - warmup) / max(1, total_epochs - warmup)
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def train(
    model: torch.nn.Module,
    train_ds,
    cfg,
    out_dir: Path,
    loss_step: Callable,            # (model, batch, device) -> scalar loss
    val_ds=None,
    val_step: Callable | None = None,
    device: str = "cuda",
    tag: str = "",
):
    """Generic training loop. Saves best (by val if available, else train) loss."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    scaler = make_scaler(cfg.amp and device == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = None
    if val_ds is not None and val_step is not None:
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

    best = float("inf")
    for epoch in range(cfg.epochs):
        lr = cosine_lr(opt, cfg.lr, epoch, cfg.epochs)
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with amp_autocast(cfg.amp and device == "cuda"):
                loss = loss_step(model, batch, device)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            n += 1

        train_loss = running / max(1, n)
        val_loss = None
        if val_loader is not None:
            model.eval()
            vs, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    with amp_autocast(cfg.amp and device == "cuda"):
                        vl = val_step(model, batch, device)
                    vs += vl.item()
                    vn += 1
            val_loss = vs / max(1, vn)

        monitor = val_loss if val_loss is not None else train_loss
        rec = {"epoch": epoch, "lr": round(lr, 8), "train_loss": train_loss,
               "val_loss": val_loss, "sec": round(time.time() - t0, 1), "tag": tag}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{tag}] epoch {epoch+1}/{cfg.epochs} "
              f"train={train_loss:.5f}" + (f" val={val_loss:.5f}" if val_loss is not None else "")
              + f" lr={lr:.2e} ({rec['sec']}s)", flush=True)

        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": vars(cfg)},
                   out_dir / "last.pth")
        if monitor < best:
            best = monitor
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "monitor": best, "cfg": vars(cfg)}, out_dir / "best.pth")

    (out_dir / "DONE").write_text(f"best={best}\n")
    return best
