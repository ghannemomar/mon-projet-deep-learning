"""Étape DVC "prepare" : split train/val/test + redimensionnement de SH17.

- test  = val officiel de SH17 (1 620 images) -> comparaison avec le benchmark publié
- val   = 10 % du train officiel, tirés avec une seed fixe (42)
- train = le reste du train officiel
Le split est fait AVANT toute augmentation (pas de fuite de données).
Les labels YOLO sont normalisés (0-1) : un redimensionnement proportionnel ne les change pas.

Lancement : `dvc repro` (ou `python -m src.data.dataset`).
"""

from __future__ import annotations

import json
import os
import random
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import yaml
from PIL import Image, ImageOps

# Ordre de sh17.yaml (alphabétique), PAS celui du README (vérifié visuellement)
NAMES = [
    "person",
    "ear",
    "ear-mufs",
    "face",
    "face-guard",
    "face-mask",
    "foot",
    "tool",
    "glasses",
    "gloves",
    "helmet",
    "hands",
    "head",
    "medical-suit",
    "shoes",
    "safety-suit",
    "safety-vest",
]

RAW = Path(os.getenv("SH17_RAW", "data/raw"))
OUT = Path("data/processed")


def read_list(path: Path) -> list[str]:
    """Les fichiers *_files.txt contiennent un nom (ou un chemin) d'image par ligne."""
    lines = path.read_text().splitlines()
    return sorted({Path(line.strip().replace("\\", "/")).name for line in lines if line.strip()})


def process(name: str, split: str, max_side: int) -> int:
    stem = Path(name).stem
    with Image.open(RAW / "images" / name) as img:
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            # draft = décodage JPEG directement à une résolution réduite (beaucoup plus rapide)
            img.draft("RGB", (int(w * scale), int(h * scale)))
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        img.save(OUT / "images" / split / f"{stem}.jpg", quality=92)
    src_lbl = RAW / "labels" / f"{stem}.txt"
    dst_lbl = OUT / "labels" / split / f"{stem}.txt"
    if src_lbl.exists():
        shutil.copy(src_lbl, dst_lbl)
        return sum(1 for line in src_lbl.read_text().splitlines() if line.strip())
    dst_lbl.touch()  # image sans objet
    return 0


def make_splits(train: list[str], test: list[str], val_frac: float, seed: int) -> dict[str, list[str]]:
    assert not set(train) & set(test), "Fuite : une image est à la fois dans train et test"
    train = sorted(train)
    random.Random(seed).shuffle(train)
    n_val = round(len(train) * val_frac)
    return {"train": sorted(train[n_val:]), "val": sorted(train[:n_val]), "test": sorted(test)}


def main() -> None:
    p = yaml.safe_load(open("params.yaml"))["prepare"]
    splits = make_splits(
        read_list(RAW / "train_files.txt"), read_list(RAW / "val_files.txt"), p["val_frac"], p["seed"]
    )
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {}
    for split, names in splits.items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        with Pool(os.cpu_count()) as pool:
            counts = pool.map(partial(process, split=split, max_side=p["max_side"]), names, chunksize=16)
        stats[split] = {"images": len(names), "instances": sum(counts)}
        print(f"{split}: {stats[split]}")

    cfg = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(NAMES)),
    }
    yaml.safe_dump(cfg, open(OUT / "sh17.yaml", "w"), sort_keys=False)
    json.dump(stats, open(OUT / "split_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
