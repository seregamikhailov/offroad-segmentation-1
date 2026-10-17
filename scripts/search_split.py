"""Подбор разбиения RUGD на train/val/test по видеопоследовательностям.

Критерии (в порядке важности):
  1. каждый класс, который есть в данных, присутствует в val и test (иначе mIoU по нему не определён);
  2. размеры выборок близки к целевым (val ~10%, test ~25% кадров);
  3. распределение классов в val/test близко к train (L1-расстояние).
Результат пишется в configs/rugd_split.json и подхватывается prepare_rugd.py.
Запуск: python scripts/search_split.py   (после prepare_rugd.py, он создаёт data/rugd/splits)
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/rugd")
K = len(json.loads((ROOT / "classes.json").read_text())["classes"])
STEP = 4  # каждый 4-й кадр для подсчёта пикселей
MIN_PIX = 2000  # класс «присутствует» в выборке, если пикселей >= MIN_PIX (в подвыборке кадров)

pairs = []
for sp in ("train", "val", "test"):
    pairs += [l.split("\t") for l in (ROOT / "splits" / f"{sp}.txt").read_text().split("\n") if l.strip()]

counts, frames = defaultdict(lambda: np.zeros(K, np.int64)), defaultdict(int)
for i, (img, mask) in enumerate(pairs):
    seq = Path(img).parent.name
    frames[seq] += 1
    if i % STEP == 0:
        m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        counts[seq] += np.bincount(m[m != 255], minlength=K)[:K]

seqs = sorted(frames)
C = np.stack([counts[s] for s in seqs]).astype(np.float64)       # S x K
F = np.array([frames[s] for s in seqs], dtype=np.float64)
present = C.sum(0) >= MIN_PIX
print("классов с данными:", int(present.sum()), "из", K)

def score(assign):  # assign: массив 0/1/2 = train/val/test
    sc, parts = 0.0, []
    for k in (0, 1, 2):
        idx = assign == k
        parts.append((C[idx].sum(0), F[idx].sum()))
    (ctr, ftr), (cva, fva), (cte, fte) = parts
    if ftr == 0 or fva == 0 or fte == 0:
        return 1e9
    miss = (cva[present] < MIN_PIX).sum() + (cte[present] < MIN_PIX).sum()
    n = F.sum()
    size_pen = abs(fva / n - 0.10) + abs(fte / n - 0.25)  # ~65% train
    ptr = ctr / ctr.sum()
    l1 = np.abs(cva / max(cva.sum(), 1) - ptr).sum() + np.abs(cte / max(cte.sum(), 1) - ptr).sum()
    return 100 * miss + 150 * size_pen + l1

rnd = np.random.default_rng(42)
best, best_s = None, 1e18
for _ in range(300000):
    a = rnd.integers(0, 3, len(seqs))
    s = score(a)
    if s < best_s:
        best, best_s = a.copy(), s

split = {s: ["train", "val", "test"][k] for s, k in zip(seqs, best)}
Path("configs").mkdir(exist_ok=True)
Path("configs/rugd_split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False))
print("score", round(best_s, 3))
for name in ("train", "val", "test"):
    ss = [s for s in seqs if split[s] == name]
    print(f"{name}: {sum(frames[s] for s in ss)} кадров — {ss}")
