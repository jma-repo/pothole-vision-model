"""Find potholes in Tesla dashcam clips and locate them with the GPS the car writes into each frame.

    python detect_tesla.py <clip.mp4 | TeslaCam folder> ... --weights models/pothole_rdd_yolo26s.pt --out results

Writes results/potholes.csv (one row per tracked pothole) plus a snapshot of each pothole.
GPS comes from the SEI metadata in Tesla dashcam video (firmware 2025.44.25+, HW3+). lat/lon is the
car's position on the last frame (with GPS) the pothole was visible, so the pothole sits a few metres ahead of it.
"""
import argparse
import csv
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from ultralytics import YOLO

POTHOLE = "D40"
TRACKER = Path(__file__).with_name("pothole_tracker.yaml")
DEFAULT_CONF = 0.25


@dataclass(frozen=True)
class FrameMeta:
    lat: float
    lon: float
    heading: float
    speed_mps: float


# Field numbers from Tesla's dashcam.proto (github.com/teslamotors/dashcam).
SEI_FIELDS = {4: "speed_mps", 11: "lat", 12: "lon", 13: "heading"}


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, i
        shift += 7


def decode_sei(buf: bytes) -> FrameMeta:
    # proto3 leaves out zero-valued fields, so anything absent is 0.
    values = dict.fromkeys(SEI_FIELDS.values(), 0.0)
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            (value,) = struct.unpack_from("<d", buf, i)
            i += 8
        elif wire == 5:
            (value,) = struct.unpack_from("<f", buf, i)
            i += 4
        elif wire == 2:
            size, i = _varint(buf, i)
            value = buf[i : i + size]
            i += size
        else:
            raise ValueError(f"unexpected protobuf wire type {wire}")
        if field in SEI_FIELDS:
            values[SEI_FIELDS[field]] = value
    return FrameMeta(**values)


def _tesla_payload(nal: bytes) -> bytes | None:
    # Layout: NAL header, SEI payload type, payload size, a run of 0x42 bytes closed by 0x69,
    # the protobuf, then the RBSP stop byte.
    i = 3
    while i < len(nal) - 1 and nal[i] == 0x42:
        i += 1
    if i >= len(nal) - 1 or nal[i] != 0x69:
        return None
    return re.sub(rb"\x00\x00\x03", b"\x00\x00", nal[i + 1 : -1])  # drop emulation-prevention bytes


def _seek_mdat(f: BinaryIO) -> int | None:
    """Position f at the mdat payload; return its end offset (None if it runs to EOF)."""
    while True:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("no mdat box")
        size, kind = struct.unpack(">I4s", header)
        header_len = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header_len = 16
        if kind == b"mdat":
            return f.tell() + size - header_len if size else None
        f.seek(size - header_len, 1)


def read_sei(path: Path) -> list[FrameMeta | None]:
    """Metadata for each frame in stream order, None for frames the car wrote none for (e.g. before pulling away).

    Tesla puts each frame's SEI right before that frame's slice, and the clips have no B-frames, so stream order
    is display order.
    """
    frames: list[FrameMeta | None] = []
    pending = None
    with path.open("rb") as f:
        end = _seek_mdat(f)
        while end is None or f.tell() < end:
            head = f.read(6)  # 4-byte NAL length, NAL header, first payload byte
            if len(head) < 6:
                break
            size = struct.unpack(">I", head[:4])[0]
            if size < 2:
                raise ValueError(f"{path}: bad NAL length {size} at byte {f.tell() - 6}")
            kind = head[4] & 0x1F
            if kind == 6 and head[5] == 5:  # SEI user-data-unregistered
                payload = _tesla_payload(head[4:] + f.read(size - 2))
                if payload is not None:
                    pending = decode_sei(payload)
                continue
            if kind in (1, 5) and head[5] & 0x80:  # slice with first_mb_in_slice == 0 starts a frame
                frames.append(pending)
                pending = None
            f.seek(size - 2, 1)
    return frames


@dataclass
class Track:
    first_frame: int
    last_frame: int
    hits: int
    best_conf: float
    best_box: list[int]
    best_frame: np.ndarray
    box_by_frame: dict[int, list[int]] = field(default_factory=dict)


def track_potholes(weights: Path, clip: Path, conf: float) -> tuple[dict[int, Track], float, int]:
    model = YOLO(weights)  # a fresh model per clip also gives a fresh tracker
    pothole_id = {name: i for i, name in model.names.items()}[POTHOLE]
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracks: dict[int, Track] = {}
    frame_idx = -1
    while cap.grab():  # every frame: near the car a pothole moves too far between every 2nd frame to match by IoU
        frame_idx += 1
        ok, frame = cap.retrieve()
        if not ok:
            raise RuntimeError(f"{clip}: could not decode frame {frame_idx}")
        boxes = model.track(
            frame, persist=True, tracker=str(TRACKER), classes=[pothole_id], conf=conf, verbose=False
        )[0].boxes
        if boxes.id is None:
            continue
        for tid, c, xyxy in zip(boxes.id.int().tolist(), boxes.conf.tolist(), boxes.xyxy.int().tolist()):
            t = tracks.get(tid)
            if t is None:
                t = tracks[tid] = Track(frame_idx, frame_idx, 0, 0.0, xyxy, frame)
            t.last_frame = frame_idx
            t.hits += 1
            t.box_by_frame[frame_idx] = xyxy
            if c > t.best_conf:
                t.best_conf, t.best_box, t.best_frame = c, xyxy, frame
    cap.release()
    return tracks, fps, frame_idx + 1


def clip_start(clip: Path) -> datetime | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", clip.name)
    return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S") if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="front-camera MP4s, or folders searched for *-front.mp4")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="detection confidence threshold")
    ap.add_argument("--min-hits", type=int, default=3, help="frames a pothole must be detected in to count")
    args = ap.parse_args()

    clips = sorted(p for i in args.inputs for p in ([i] if i.is_file() else i.rglob("*-front.mp4")))
    if not clips:
        raise SystemExit("no clips found")
    snap_dir = args.out / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for clip in clips:
        metas = read_sei(clip)
        tracks, fps, n_frames = track_potholes(args.weights, clip, args.conf)
        start = clip_start(clip)
        kept = {tid: t for tid, t in tracks.items() if t.hits >= args.min_hits}
        n_gps = sum(m is not None for m in metas)
        gps_note = f"GPS on {n_gps} frames" if n_gps else "NO GPS (needs firmware 2025.44.25+, HW3+)"
        print(f"{clip.name}: {n_frames} frames, {gps_note}, {len(kept)} potholes ({len(tracks)} raw tracks)")
        for tid, t in kept.items():
            meta = next((m for m in reversed(metas[t.first_frame : t.last_frame + 1]) if m is not None), None)
            x1, y1, x2, y2 = t.best_box
            snap = t.best_frame.copy()
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(snap, f"pothole {t.best_conf:.2f}", (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            snap_path = snap_dir / f"{clip.stem}_{tid}.jpg"
            cv2.imwrite(str(snap_path), snap)
            rows.append({
                "clip": clip.name,
                "track_id": tid,
                "seen_at": (start + timedelta(seconds=t.last_frame / fps)).isoformat(sep=" ") if start else "",
                "first_s": round(t.first_frame / fps, 2),
                "last_s": round(t.last_frame / fps, 2),
                "hits": t.hits,
                "max_conf": round(t.best_conf, 3),
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "lat": round(meta.lat, 7) if meta else "",
                "lon": round(meta.lon, 7) if meta else "",
                "heading_deg": round(meta.heading, 1) if meta else "",
                "speed_kmh": round(meta.speed_mps * 3.6, 1) if meta else "",
                "snapshot": snap_path.relative_to(args.out).as_posix(),
            })

    csv_path = args.out / "potholes.csv"
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["clip"])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)
    print(f"{len(rows)} potholes -> {csv_path}")


if __name__ == "__main__":
    main()
