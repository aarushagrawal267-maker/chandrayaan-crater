"""Train CraterUNet on procedurally generated lunar scenes (CPU friendly).

Example::

    python -m lunarcrater train --epochs 16 --train-size 1600

Scenes are generated deterministically from seeds, so no dataset has to be downloaded and
the run is reproducible.  Training, validation and test scenes use disjoint seed ranges.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .detect import DEFAULT_MODEL, standardize
from .model import CraterUNet, count_parameters, save_model
from .synthetic import generate_scene

TRAIN_SEED0 = 0
VAL_SEED0 = 10_000_000
TEST_SEED0 = 20_000_000


class SyntheticCraterDataset(Dataset):
    def __init__(self, n: int, seed0: int, size: int = 160, augment: bool = False):
        self.n, self.seed0, self.size, self.augment = n, seed0, size, augment

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        sc = generate_scene(self.size, seed=self.seed0 + i)
        img, mask = sc.image, sc.rim_mask
        if self.augment:
            rng = np.random.default_rng((self.seed0 + i) * 7 + 3)
            k = int(rng.integers(0, 4))
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
        x = standardize(np.ascontiguousarray(img))[None]
        y = np.ascontiguousarray(mask, dtype=np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(y)


def loss_fn(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 4.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight))
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


def evaluate_loss(model, loader) -> tuple[float, float]:
    model.eval()
    total, iou_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total += loss_fn(logits, y).item() * len(x)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            inter = (pred * y).sum(dim=(1, 2, 3))
            union = ((pred + y) > 0).float().sum(dim=(1, 2, 3)).clamp(min=1)
            iou_sum += (inter / union).sum().item()
            n += len(x)
    return total / n, iou_sum / n


def train(
    epochs: int = 12,
    train_size: int = 1200,
    val_size: int = 100,
    batch_size: int = 8,
    lr: float = 2e-3,
    base: int = 16,
    size: int = 160,
    out: str | Path = DEFAULT_MODEL,
    workers: int = 3,
    init_from: str | Path | None = None,
    schedule: str = "onecycle",
    log=print,
) -> dict:
    """Train from scratch, or continue from ``init_from`` (e.g. a low-LR cosine fine-tune).

    ``schedule`` is ``onecycle`` (warm-up then anneal) or ``cosine`` (anneal from ``lr``)."""
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = CraterUNet(base=base)
    prior_history: list[dict] = []
    if init_from:
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        model = CraterUNet(base=ckpt.get("base", base), depth=ckpt.get("depth", 4))
        model.load_state_dict(ckpt["state_dict"])
        prior_history = list(ckpt.get("extra", {}).get("history", []))
        log(f"continuing from {init_from} ({len(prior_history)} earlier epochs)")
    log(f"CraterUNet: {count_parameters(model):,} trainable parameters")
    train_dl = DataLoader(
        SyntheticCraterDataset(train_size, TRAIN_SEED0, size=size, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    val_dl = DataLoader(SyntheticCraterDataset(val_size, VAL_SEED0, size=size), batch_size=16, num_workers=workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(train_dl)
    if schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    history = list(prior_history)
    start_epoch = len(prior_history)
    best = float("inf") if not init_from else evaluate_loss(model, val_dl)[0]
    if init_from:
        log(f"starting val loss {best:.4f}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(start_epoch + 1, start_epoch + epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(x)
            n += len(x)
        val_loss, val_iou = evaluate_loss(model, val_dl)
        rec = {"epoch": ep, "stage": "fine-tune" if init_from else "train", "lr_schedule": f"{schedule} lr={lr:g}", "images_per_epoch": train_size, "train_loss": round(running / n, 4), "val_loss": round(val_loss, 4), "val_rim_iou": round(val_iou, 4), "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        log(json.dumps(rec))
        if val_loss < best:
            best = val_loss
            save_model(model, out, extra={"history": history, "train_size": train_size, "epochs": len(history), "base": model.base, "train_image_size": size})
    if not out.exists() and init_from:  # fine-tune never beat its starting point
        import shutil

        shutil.copy(init_from, out)
    # store full history with the best checkpoint
    ckpt = torch.load(out, weights_only=False)
    ckpt["extra"]["history"] = history
    torch.save(ckpt, out)
    (out.parent / "training_history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "best_val_loss": best, "model_path": str(out)}
