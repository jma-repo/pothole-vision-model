"""Build a fine-tuning set from reviewed Tesla detections, mixed with RDD2022 so the model keeps what it learned there.

    python build_finetune.py --labels pothole_labels.json --clips CLIPS --weights W --rdd data/rdd2022_yolo --test 2026-11-17_16 2025-12-14 --out data/finetune

Clips whose names start with a --test prefix are left out, for evaluation with eval_labels.py. Up to four frames of
each reviewed track become training images, plus a few plain-road frames from each clip:
- confirmed potholes are labelled as potholes; their frames are picked where they are close enough to see clearly;
- rejected detections stay unlabelled, so they teach the model what is not a pothole;
- plain-road frames (no pothole-like prediction at all) show what wet, snowy or night road looks like without one;
- unsure or unreviewed pothole detections are greyed out, so they teach nothing either way;
- cracks get the current model's own predictions as labels, so fine-tuning does not unlearn them.
All of RDD2022's training images are mixed in for the same reason.
"""
import argparse
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from build_map import read_labels
from detect_tesla import DEFAULT_CONF, POTHOLE, track_potholes

FRAMES_PER_TRACK = 4
MIN_POTHOLE_BOX_H = 24
BACKGROUND_PER_CLIP = 5
BACKGROUND_CANDIDATES_PER_CLIP = 20
TESLA_REPEATS = 5
CRACK_CONF = 0.4


def iou(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def yolo_line(cls: int, box: list[float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def spread(frames: list[int], k: int) -> list[int]:
    if len(frames) <= k:
        return frames
    return [frames[round(i * (len(frames) - 1) / (k - 1))] for i in range(k)]


def frames_at(clip: Path, wanted: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(clip))
    last = max(wanted, default=-1)
    idx = -1
    while idx < last and cap.grab():
        idx += 1
        if idx in wanted:
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f"{clip}: could not decode frame {idx}")
            yield idx, frame
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True, help="the model that produced the reviewed detections")
    ap.add_argument("--rdd", type=Path, required=True, help="prepare_rdd.py output folder (train.txt, val.txt)")
    ap.add_argument("--test", nargs="+", required=True, help="clip-name prefixes held out for evaluation")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    labels = read_labels(args.labels)
    clips = {p.name: p for p in args.clips.rglob("*-front.mp4")}
    reviewed = sorted({i.split("#")[0] for i, lab in labels.items() if lab in ("yes", "no")})
    train_clips = [c for c in reviewed if not c.startswith(tuple(args.test))]
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    pothole_id = {n: i for i, n in model.names.items()}[POTHOLE]

    images, counts = [], Counter()
    rng = random.Random(0)
    for name in train_clips:
        tracks, fps, n_frames = track_potholes(args.weights, clips[name], DEFAULT_CONF)
        status = {tid: labels.get(f"{name}#{tid}") for tid in tracks}
        wanted: set[int] = set()
        for tid, t in tracks.items():
            frames = sorted(t.box_by_frame)
            if status[tid] == "yes":
                frames = [f for f in frames if t.box_by_frame[f][3] - t.box_by_frame[f][1] >= MIN_POTHOLE_BOX_H]
            if status[tid] in ("yes", "no"):
                wanted.update(spread(frames, FRAMES_PER_TRACK))
        margin = round(fps)
        busy = {f for t in tracks.values() for f in range(t.first_frame - margin, t.last_frame + margin)}
        free = [f for f in range(n_frames) if f not in busy]
        background_candidates = set(rng.sample(free, k=min(len(free), BACKGROUND_CANDIDATES_PER_CLIP)))
        kept_background = 0
        for idx, frame in frames_at(clips[name], wanted | background_candidates):
            h, w = frame.shape[:2]
            present = {tid: t.box_by_frame[idx] for tid, t in tracks.items() if idx in t.box_by_frame}
            lines = [yolo_line(pothole_id, b, w, h) for tid, b in present.items() if status[tid] == "yes"]
            reviewed_boxes = [b for tid, b in present.items() if status[tid] in ("yes", "no")]
            grey = [b for tid, b in present.items() if status[tid] not in ("yes", "no")]
            pred = model.predict(frame, conf=0.1, verbose=False)[0].boxes
            if idx in background_candidates:
                looks_like_pothole = pothole_id in pred.cls.int().tolist()
                if looks_like_pothole or kept_background >= BACKGROUND_PER_CLIP:
                    continue
                kept_background += 1
                counts["plain-road frames"] += 1
            for c, k, b in zip(pred.conf.tolist(), pred.cls.int().tolist(), pred.xyxy.tolist()):
                if k == pothole_id:
                    if c >= DEFAULT_CONF and not any(iou(b, r) > 0.3 for r in reviewed_boxes):
                        grey.append(b)
                elif c >= CRACK_CONF:
                    lines.append(yolo_line(k, b, w, h))
            for x1, y1, x2, y2 in (map(int, b) for b in grey):
                region = frame[max(y1, 0):y2, max(x1, 0):x2]
                if region.size:
                    region[:] = region.reshape(-1, 3).mean(axis=0)
            stem = f"{Path(name).stem}_{idx:05d}"
            cv2.imwrite(str(args.out / "images" / f"{stem}.jpg"), frame)
            (args.out / "labels" / f"{stem}.txt").write_text("\n".join(lines))
            images.append((args.out / "images" / f"{stem}.jpg").resolve().as_posix())
            counts["frames"] += 1
            counts["frames with a pothole"] += any(status[t] == "yes" for t in present)
            counts["frames with a rejected detection"] += any(status[t] == "no" for t in present)
        print(f"{name}: {counts['frames']} frames so far", flush=True)

    rdd = (args.rdd / "train.txt").read_text().split()
    (args.out / "train.txt").write_text("\n".join(images * TESLA_REPEATS + rdd))
    names = "\n".join(f"  {i}: {n}" for i, n in model.names.items())
    (args.out / "finetune.yaml").write_text(
        f"train: {(args.out / 'train.txt').resolve().as_posix()}\n"
        f"val: {(args.rdd / 'val.txt').resolve().as_posix()}\nnames:\n{names}\n")
    print(f"{len(train_clips)} training clips, {dict(counts)}; "
          f"train list: {len(images)} Tesla frames x{TESLA_REPEATS} + {len(rdd)} RDD images")


if __name__ == "__main__":
    main()
