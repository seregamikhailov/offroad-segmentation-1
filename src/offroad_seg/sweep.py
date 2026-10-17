"""Run a list of experiments defined in a YAML sweep file.

python -m offroad_seg.sweep --sweep configs/sweeps/screening.yaml [--group arch] [--only name1,name2] [--dry-run]

Sweep file format:
  name: screening                # runs go to runs/<name>/<experiment>/
  base: configs/baseline.yaml    # base config
  common:                        # overrides applied to every experiment (dotted keys)
    train.epochs: 20
  experiments:
    - name: unet_r34
      group: arch
      set: {model.arch: Unet}

Finished experiments (summary.json present) are skipped, so an interrupted sweep can simply be restarted.
"""
import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml


def set_dotted(cfg: dict, key: str, value):
    node = cfg
    *parents, last = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--group", help="run only experiments of this group")
    ap.add_argument("--only", help="comma-separated experiment names")
    ap.add_argument("--limit", type=int, help="limit frames per split (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.sweep))
    base = yaml.safe_load(open(spec["base"]))
    only = set(args.only.split(",")) if args.only else None
    root = Path(args.runs_dir) / spec["name"]

    todo = []
    for exp in spec["experiments"]:
        if args.group and exp.get("group") != args.group:
            continue
        if only and exp["name"] not in only:
            continue
        todo.append(exp)
    print(f"{len(todo)} experiment(s) selected from {args.sweep}")

    for i, exp in enumerate(todo, 1):
        out = root / exp["name"]
        if (out / "summary.json").exists():
            print(f"[{i}/{len(todo)}] {exp['name']}: already done, skipping")
            continue
        cfg = copy.deepcopy(base)
        for k, v in {**spec.get("common", {}), **exp.get("set", {})}.items():
            set_dotted(cfg, k, v)
        cfg["output_dir"] = str(out)
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "sweep_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cmd = [sys.executable, "-m", "offroad_seg.train", "--config", str(cfg_path), "--output-dir", str(out)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"[{i}/{len(todo)}] {exp['name']} ({exp.get('group', '-')}) -> {out}")
        if args.dry_run:
            print("   ", " ".join(cmd))
            continue
        t0 = time.time()
        with open(out / "train.log", "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if code == 0 else f"FAILED (exit {code}, see {out/'train.log'})"
        print(f"    {status} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
