"""Score a detection run against the hand-reviewed detections of an earlier run, on held-out clips.

    python eval_labels.py --labels pothole_labels.json --reviewed results --new results_new --test 2026-11-17_16 2025-12-14

A reviewed detection counts as found again when the new run has a detection in the same clip whose time window
overlaps it (give or take 0.5 s) and, if both have GPS, whose GPS position is within 20 m. New detections that match
nothing reviewed are counted separately: nobody has judged them yet.
"""
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from build_map import read_labels

SLACK_S, NEAR_M = 0.5, 20.0


@dataclass(frozen=True)
class Detection:
    id: str  # clip#track_id, the key the review labels use
    clip: str
    first_s: float
    last_s: float
    car_lat_lon: tuple[float, float] | None


def load(results: Path, prefixes: tuple[str, ...]) -> list[Detection]:
    with (results / "potholes.csv").open(newline="", encoding="utf-8") as f:
        return [
            Detection(
                id=f"{r['clip']}#{r['track_id']}",
                clip=r["clip"],
                first_s=float(r["first_s"]),
                last_s=float(r["last_s"]),
                car_lat_lon=(float(r["lat"]), float(r["lon"])) if r["lat"] else None,
            )
            for r in csv.DictReader(f)
            if r["clip"].startswith(prefixes)
        ]


def same_pothole(a: Detection, b: Detection) -> bool:
    if a.clip != b.clip:
        return False
    if a.first_s > b.last_s + SLACK_S or b.first_s > a.last_s + SLACK_S:
        return False
    if a.car_lat_lon is None or b.car_lat_lon is None:
        return True
    (lat_a, lon_a), (lat_b, lon_b) = a.car_lat_lon, b.car_lat_lon
    dy = (lat_a - lat_b) * 111_320
    dx = (lon_a - lon_b) * 111_320 * math.cos(math.radians(lat_a))
    return math.hypot(dx, dy) < NEAR_M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--reviewed", type=Path, required=True, help="the run the labels belong to")
    ap.add_argument("--new", type=Path, required=True, help="the run to score")
    ap.add_argument("--test", nargs="+", required=True, help="clip-name prefixes of the held-out clips")
    args = ap.parse_args()

    labels = read_labels(args.labels)
    prefixes = tuple(args.test)
    reviewed = [d for d in load(args.reviewed, prefixes) if labels.get(d.id)]
    new = load(args.new, prefixes)
    found = {lab: [0, 0] for lab in ("yes", "no", "unsure")}
    for r in reviewed:
        lab = labels[r.id]
        found[lab][1] += 1
        found[lab][0] += any(same_pothole(r, n) for n in new)
    unmatched = sum(not any(same_pothole(n, r) for r in reviewed) for n in new)
    (yes_hit, yes_n), (no_hit, no_n) = found["yes"], found["no"]
    print(f"confirmed potholes found again: {yes_hit}/{yes_n}")
    print(f"rejected detections raised again: {no_hit}/{no_n}")
    print(f"unsure detections raised again: {found['unsure'][0]}/{found['unsure'][1]}")
    print(f"new detections nobody has reviewed: {unmatched}")
    print(f"precision on reviewed detections: {yes_n / (yes_n + no_n):.0%} before, "
          f"{yes_hit / max(yes_hit + no_hit, 1):.0%} after")


if __name__ == "__main__":
    main()
