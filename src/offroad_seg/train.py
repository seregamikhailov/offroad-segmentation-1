"""Обучение:  python -m offroad_seg.train --config configs/baseline.yaml [--limit 64 --epochs 1]"""
import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SegDataset, class_frequencies, load_classes
from .losses import build_loss, inverse_frequency_weights
from .metrics import ConfusionMatrix
from .models import build_model, count_params, pick_device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index, loss_fn=None):
    model.eval()
    cm, total_loss, n = ConfusionMatrix(num_classes, ignore_index), 0.0, 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_fn is not None:
            total_loss += loss_fn(logits, y).item() * x.size(0)
            n += x.size(0)
        cm.update(logits.argmax(1), y)
    res = cm.compute()
    res["loss"] = total_loss / max(n, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="ограничить число кадров (отладка)")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")

    tc, dc, mc = cfg["train"], cfg["data"], cfg["model"]
    seed_everything(tc["seed"])
    device = pick_device(tc["device"])
    classes, ignore = load_classes(dc["root"])
    num_classes = len(classes)
    size = tuple(dc["image_size"])

    train_ds = SegDataset(dc["root"], "train", size, train=True, limit=args.limit)
    val_ds = SegDataset(dc["root"], "val", size, train=False, limit=args.limit)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, tc["batch_size"], shuffle=True, num_workers=dc["num_workers"],
                          drop_last=True, pin_memory=pin, persistent_workers=dc["num_workers"] > 0)
    val_dl = DataLoader(val_ds, tc["batch_size"], shuffle=False, num_workers=dc["num_workers"], pin_memory=pin)

    model = build_model(mc["arch"], mc["encoder"], num_classes, mc["encoder_weights"]).to(device)
    weights = None
    if tc.get("class_weights"):
        freq = class_frequencies(dc["root"], "train", num_classes, ignore)
        weights = inverse_frequency_weights(freq).to(device)
    loss_fn = build_loss(tc["loss"], ignore, weights)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["epochs"] * max(len(train_dl), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=total_steps, pct_start=0.1)
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={device} | train={len(train_ds)} val={len(val_ds)} | classes={num_classes} | "
          f"params={count_params(model)/1e6:.1f}M | amp={use_amp}")

    log_path, best = out / "log.csv", -1.0
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc", "sec"])

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{tc['epochs']}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        res = evaluate(model, val_dl, device, num_classes, ignore, loss_fn)
        tl, sec = run / max(len(train_dl), 1), time.time() - t0
        print(f"epoch {epoch}: train_loss={tl:.4f} val_loss={res['loss']:.4f} "
              f"mIoU={res['mIoU']:.4f} acc={res['pixel_acc']:.4f} ({sec:.0f}s)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tl:.5f}", f"{res['loss']:.5f}", f"{res['mIoU']:.5f}",
                                    f"{res['pixel_acc']:.5f}", f"{sec:.1f}"])
        if res["mIoU"] > best:
            best = res["mIoU"]
            torch.save({"model": model.state_dict(), "config": cfg, "classes": classes,
                        "epoch": epoch, "val_mIoU": best}, out / "best.pt")
            per_class = {c["name"]: (None if np.isnan(v) else round(float(v), 4))
                         for c, v in zip(classes, res["iou_per_class"])}
            (out / "best_iou_per_class.json").write_text(json.dumps(per_class, ensure_ascii=False, indent=2))
    print(f"best val mIoU = {best:.4f} -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
