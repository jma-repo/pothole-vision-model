"""Convert RDD2022 Pascal VOC annotations into a YOLO detection dataset.

Labels are written next to the images (<Country>/train/labels/*.txt), where Ultralytics
looks for them; the split lists and the dataset yaml go to --out.
"""
import argparse
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

CLASSES = ["D00", "D10", "D20", "D40"]  # longitudinal crack, transverse crack, alligator crack, pothole
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Box:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float


def read_voc(xml_path: Path) -> list[Box]:
    boxes = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        bnd = obj.find("bndbox")
        coords = (float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append(Box(obj.findtext("name", "").strip(), *coords))
    return boxes


def to_yolo(box: Box, width: int, height: int) -> str | None:
    """YOLO label line for a known-class box, or None if it clips to nothing inside the image."""
    x1, x2 = max(0.0, min(box.x1, box.x2)), min(float(width), max(box.x1, box.x2))
    y1, y2 = max(0.0, min(box.y1, box.y2)), min(float(height), max(box.y1, box.y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{CLASS_ID[box.name]} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def split_for(stem: str) -> str:
    bucket = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 10
    return {0: "test", 1: "val"}.get(bucket, "train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="extracted RDD2022 folder holding <Country>/train")
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="where the split lists and rdd2022.yaml go")
    args = ap.parse_args()

    splits = defaultdict(list)
    counts = defaultdict(Counter)
    dropped = Counter()
    for country in args.countries:
        train_dir = args.root / country / "train"
        xmls = sorted((train_dir / "annotations" / "xmls").glob("*.xml"))
        if not xmls:
            raise SystemExit(f"no annotations under {train_dir}")
        label_dir = train_dir / "labels"
        label_dir.mkdir(exist_ok=True)
        for xml_path in xmls:
            img_path = train_dir / "images" / f"{xml_path.stem}.jpg"
            with Image.open(img_path) as im:
                width, height = im.size
            lines = []
            for box in read_voc(xml_path):
                if box.name not in CLASS_ID:
                    dropped[box.name] += 1
                elif (line := to_yolo(box, width, height)) is None:
                    dropped[f"{box.name} zero-size"] += 1
                else:
                    lines.append(line)
            (label_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines))
            split = split_for(xml_path.stem)
            splits[split].append(img_path.resolve().as_posix())
            counts[split].update(CLASSES[int(line.split()[0])] for line in lines)
            counts[split]["images"] += 1
            counts[split]["background_images"] += not lines

    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (args.out / f"{split}.txt").write_text("\n".join(splits[split]) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (args.out / "rdd2022.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n{names}\n"
    )

    for split in SPLITS:
        print(split, dict(sorted(counts[split].items())))
    print("dropped labels:", dict(dropped))


if __name__ == "__main__":
    main()
