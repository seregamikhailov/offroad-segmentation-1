"""Распаковка RUGD и подготовка данных.

Что делает:
  1. распаковывает архивы из data/raw в data/raw/extracted;
  2. читает таблицу цветов классов (colormap .txt из архива аннотаций);
  3. переводит цветные маски в одноканальные маски с id класса (void -> 255);
  4. делит кадры на train/val/test ПО ПОСЛЕДОВАТЕЛЬНОСТЯМ (соседние кадры одного
     видео почти одинаковы, случайное разбиение по кадрам даёт утечку);
  5. пишет списки data/rugd/splits/*.txt и data/rugd/classes.json.

Запуск:  python scripts/prepare_rugd.py
"""
import argparse
import json
import random
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE = 255


def unzip(zip_path: Path, out_dir: Path):
    marker = out_dir / f".done_{zip_path.stem}"
    if marker.exists():
        return
    print(f"Распаковка {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    marker.touch()


def read_colormap(path: Path):
    """Строки вида: '<id> <name> <r> <g> <b>' -> [(id, name, (r, g, b)), ...]."""
    items = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        cid, name, (r, g, b) = int(parts[0]), " ".join(parts[1:-3]), map(int, parts[-3:])
        items.append((cid, name, (r, g, b)))
    return sorted(items)


def encode(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/rugd")
    ap.add_argument("--val-frac", type=float, default=0.15, help="доля последовательностей")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", default="configs/rugd_split.json")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    ext = raw / "extracted"
    ext.mkdir(parents=True, exist_ok=True)
    for z in sorted(raw.glob("*.zip")):
        unzip(z, ext)

    cmap_files = [p for p in ext.rglob("*.txt") if "colormap" in p.name.lower()]
    if not cmap_files:
        raise SystemExit("Не найден файл colormap (*.txt) в архиве аннотаций")
    classes = read_colormap(cmap_files[0])
    print(f"Colormap: {cmap_files[0].name}, записей: {len(classes)}")

    # void (id 0) -> IGNORE, остальные сдвигаем на -1: train_id = rugd_id - 1
    void_ids = {cid for cid, name, _ in classes if name.lower() == "void"}
    train_classes = [(cid, name, rgb) for cid, name, rgb in classes if cid not in void_ids]
    keys = np.array([(r << 16) | (g << 8) | b for _, _, (r, g, b) in train_classes], dtype=np.uint32)
    order = np.argsort(keys)
    sorted_keys = keys[order]

    ann_root = next((p for p in ext.rglob("*") if p.is_dir() and p.name.lower() == "rugd_annotations"), None)
    img_root = next((p for p in ext.rglob("*") if p.is_dir() and p.name.lower().startswith("rugd_frames")), None)
    if ann_root is None or img_root is None:
        raise SystemExit(f"Не найдены каталоги кадров/аннотаций в {ext}")

    mask_out = out / "masks"
    mask_out.mkdir(parents=True, exist_ok=True)
    pairs, unknown_px, total_px = [], 0, 0
    ann_files = sorted(ann_root.rglob("*.png"))
    for ann in tqdm(ann_files, desc="Конвертация масок"):
        rel = ann.relative_to(ann_root)
        img = img_root / rel
        if not img.exists():
            continue
        code = encode(np.array(Image.open(ann).convert("RGB")))
        pos = np.searchsorted(sorted_keys, code)
        pos = np.clip(pos, 0, len(sorted_keys) - 1)
        hit = sorted_keys[pos] == code
        mask = np.full(code.shape, IGNORE, dtype=np.uint8)
        mask[hit] = order[pos[hit]]
        # непустой цвет, которого нет в colormap, и не void -> считаем
        non_void = code != 0
        unknown_px += int((non_void & ~hit).sum())
        total_px += code.size
        dst = mask_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(dst)
        pairs.append((rel.parent.name, str(img.resolve()), str(dst.resolve())))

    print(f"Пар кадр-маска: {len(pairs)}; пикселей с неизвестным цветом: {unknown_px} ({100*unknown_px/max(total_px,1):.4f}%)")

    seqs = sorted({p[0] for p in pairs})
    split_json = Path(args.split_json)
    if split_json.exists():  # разбиение, подобранное scripts/search_split.py
        split_of = json.loads(split_json.read_text())
        print(f"Разбиение из {split_json}")
    else:  # запасной вариант: случайное по последовательностям
        rnd = random.Random(args.seed)
        rnd.shuffle(seqs)
        n_test = max(1, round(len(seqs) * args.test_frac))
        n_val = max(1, round(len(seqs) * args.val_frac))
        split_of = {}
        for i, s in enumerate(seqs):
            split_of[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    (out / "splits").mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        rows = [p for p in pairs if split_of[p[0]] == name]
        (out / "splits" / f"{name}.txt").write_text("\n".join(f"{i}\t{m}" for _, i, m in rows) + "\n")
        print(f"{name}: {len(rows)} кадров, последовательности: {sorted(s for s in seqs if split_of[s]==name)}")

    meta = {
        "ignore_index": IGNORE,
        "classes": [{"id": i, "name": n, "rgb": list(rgb)} for i, (_, n, rgb) in enumerate(train_classes)],
    }
    (out / "classes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Классов (без void): {len(train_classes)} -> {out/'classes.json'}")


if __name__ == "__main__":
    main()
