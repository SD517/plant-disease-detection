"""Stratified, group-aware train / validation / test split.

PlantVillage ships some versions with augmented copies of the same photo
(e.g. ``..._flipTB.JPG``, ``..._90deg.JPG``). If one copy lands in training and
another in the test set, the test accuracy is inflated. To prevent this we
group files that share the same original image id and keep each group inside a
single split. The split is done per class (stratified) so every disease appears
in all three sets.
"""
from __future__ import annotations

import csv
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_AUG_SUFFIX = re.compile(r"(_flip(TB|LR)|_\d+deg|_new\d+deg\w*|_Scale\w*|_Cropped\w*)+$", re.I)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def find_class_root(raw: Path) -> Path:
    """Descend through wrapper folders (e.g. ``plantvillage dataset/color``)
    until the folder that directly contains one sub-folder per class."""
    cur = Path(raw)
    if not cur.exists():
        raise FileNotFoundError(f"Raw data folder not found: {cur}")
    for _ in range(5):
        subs = [p for p in cur.iterdir() if p.is_dir()]
        names = {p.name.lower(): p for p in subs}
        if "color" in names:                      # prefer RGB images
            cur = names["color"]
            continue
        has_images = any(p.suffix.lower() in IMG_EXT for p in cur.iterdir() if p.is_file())
        if len(subs) == 1 and not has_images:     # single wrapper folder
            cur = subs[0]
            continue
        break
    return cur


def group_key(filename: str) -> str:
    """Id shared by an original photo and all its augmented copies."""
    stem = Path(filename).stem
    stem = _AUG_SUFFIX.sub("", stem)
    if "___" in stem:                              # '<uuid>___<camera code> <n>'
        return stem.split("___")[0]
    return stem


def split_dataset(class_root: Path, out_dir: Path,
                  ratios=(0.70, 0.15, 0.15), seed: int = 42,
                  mode: str = "copy", clean: bool = False) -> dict:
    """Create ``out_dir/{train,val,test}/<class>/`` and a ``split_manifest.csv``.

    mode: 'copy' (default) or 'link' (hard links, no extra disk space).
    Returns a summary dict {split: {class: n_images}}.
    """
    class_root, out_dir = Path(class_root), Path(out_dir)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")
    if out_dir.exists() and clean:
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty. Use --clean to rebuild it.")

    classes = sorted(p for p in class_root.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found in {class_root}")

    rng = random.Random(seed)
    summary = {"train": {}, "val": {}, "test": {}}
    manifest = []
    for cdir in classes:
        files = list_images(cdir)
        if not files:
            continue
        groups = defaultdict(list)
        for f in files:
            groups[group_key(f.name)].append(f)
        keys = sorted(groups)
        rng.shuffle(keys)

        n = len(files)
        t_train, t_val = ratios[0] * n, (ratios[0] + ratios[1]) * n
        seen = 0
        assigned = {"train": [], "val": [], "test": []}
        for k in keys:
            split = "train" if seen < t_train else "val" if seen < t_val else "test"
            assigned[split].extend(groups[k])
            seen += len(groups[k])

        for split, items in assigned.items():
            dest = out_dir / split / cdir.name
            dest.mkdir(parents=True, exist_ok=True)
            for src in items:
                dst = dest / src.name
                if mode == "link":
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
                manifest.append((split, cdir.name, src.name))
            summary[split][cdir.name] = len(items)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "split_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "class", "file"])
        w.writerows(manifest)
    return summary


def count_images(split_dir: Path) -> dict[str, int]:
    """{class_name: number_of_images} for one split folder."""
    return {p.name: len(list_images(p)) for p in sorted(Path(split_dir).iterdir()) if p.is_dir()}
