"""Collect summary.json files of all runs into one table.

python -m offroad_seg.aggregate --runs runs/screening [--out results]
Writes <out>/results.csv, <out>/per_class_iou.csv and prints a Markdown table sorted by val mIoU.
"""
import argparse
import csv
import json
from pathlib import Path

COLS = ["name", "arch", "encoder", "encoder_weights", "loss", "class_weights", "image_size",
        "lr", "seed", "params_M", "best_epoch", "epochs_run", "best_val_mIoU",
        "final_val_mIoU", "mean_epoch_sec", "total_train_min", "device"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.runs).rglob("summary.json")):
        s = json.loads(p.read_text())
        s["run_dir"] = str(p.parent)
        test = p.parent / "eval_test.json"
        s["test_mIoU"] = json.loads(test.read_text())["mIoU"] if test.exists() else None
        rows.append(s)
    if not rows:
        raise SystemExit(f"no summary.json under {args.runs}")
    rows.sort(key=lambda r: -r["best_val_mIoU"])

    out = Path(args.out or args.runs)
    out.mkdir(parents=True, exist_ok=True)
    cols = COLS + ["test_mIoU", "run_dir"]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])

    class_names = list(rows[0]["iou_per_class"])
    with open(out / "per_class_iou.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name"] + class_names)
        for r in rows:
            w.writerow([r["name"]] + [r["iou_per_class"].get(c) for c in class_names])

    print("| run | arch | encoder | loss | params, M | val mIoU | best ep | min/epoch |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lo = r["loss"] + ("+w" if r["class_weights"] else "")
        print(f"| {r['name']} | {r['arch']} | {r['encoder']} | {lo} | {r['params_M']} | "
              f"{100*r['best_val_mIoU']:.1f} | {r['best_epoch']} | {r['mean_epoch_sec']/60:.1f} |")
    print(f"\n{len(rows)} runs -> {out/'results.csv'}, {out/'per_class_iou.csv'}")


if __name__ == "__main__":
    main()
