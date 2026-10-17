"""Pre-flight check for a sweep: builds every model (downloading pretrained weights into the cache)
and runs one forward/backward pass, so that a broken experiment fails now and not hours into the sweep.

python scripts/check_sweep.py --sweep configs/sweeps/screening.yaml [--device cuda]
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offroad_seg.losses import build_loss  # noqa: E402
from offroad_seg.models import build_model, count_params, pick_device  # noqa: E402
from offroad_seg.sweep import set_dotted  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--classes", type=int, default=24)
    ap.add_argument("--no-weights", action="store_true", help="skip pretrained weights (no downloads)")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.sweep))
    base = yaml.safe_load(open(spec["base"]))
    dev = pick_device(args.device)
    bad = 0
    print(f"{'experiment':26s} {'params,M':>9s} {'fwd+bwd,s':>10s}  status")
    for exp in spec["experiments"]:
        cfg = copy.deepcopy(base)
        for k, v in {**spec.get("common", {}), **exp.get("set", {})}.items():
            set_dotted(cfg, k, v)
        try:
            mc, tc = cfg["model"], cfg["train"]
            weights = None if args.no_weights else mc["encoder_weights"]
            model = build_model(mc["arch"], mc["encoder"], args.classes, weights).to(dev).train()
            loss = build_loss(tc["loss"], cfg["data"]["ignore_index"], None).to(dev)
            h, w = cfg["data"]["image_size"]
            x = torch.randn(2, 3, h, w, device=dev)
            y = torch.randint(0, args.classes, (2, h, w), device=dev)
            t0 = time.time()
            loss(model(x), y).backward()
            print(f"{exp['name']:26s} {count_params(model)/1e6:9.1f} {time.time()-t0:10.2f}  ok")
        except Exception as e:  # report and continue with the rest
            bad += 1
            print(f"{exp['name']:26s} {'-':>9s} {'-':>10s}  FAILED: {type(e).__name__}: {str(e)[:100]}")
    print(f"\n{len(spec['experiments']) - bad}/{len(spec['experiments'])} experiments OK")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
