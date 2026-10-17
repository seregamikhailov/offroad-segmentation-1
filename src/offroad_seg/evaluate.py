"""Evaluate a trained checkpoint on a split (use `test` only for final, selected models).

python -m offroad_seg.evaluate --weights runs/baseline/best.pt --split test
Writes <run_dir>/eval_<split>.json with mIoU, pixel accuracy and per-class IoU.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SegDataset
from .models import build_model, pick_device
from .train import evaluate, per_class_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data-root", help="override data.root from the checkpoint config")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg, classes = ckpt["config"], ckpt["classes"]
    root = args.data_root or cfg["data"]["root"]
    ignore = cfg["data"]["ignore_index"]
    device = pick_device(args.device)

    model = build_model(cfg["model"]["arch"], cfg["model"]["encoder"], len(classes), None)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    ds = SegDataset(root, args.split, tuple(cfg["data"]["image_size"]), train=False)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    res = evaluate(model, dl, device, len(classes), ignore)

    out = {
        "weights": str(args.weights), "split": args.split, "frames": len(ds),
        "checkpoint_epoch": ckpt.get("epoch"), "checkpoint_val_mIoU": ckpt.get("val_mIoU"),
        "mIoU": round(res["mIoU"], 5), "pixel_acc": round(res["pixel_acc"], 5),
        "iou_per_class": per_class_dict(classes, res["iou_per_class"]),
    }
    path = Path(args.weights).parent / f"eval_{args.split}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"{args.split}: mIoU={out['mIoU']:.4f} pixel_acc={out['pixel_acc']:.4f} ({len(ds)} frames) -> {path}")


if __name__ == "__main__":
    main()
