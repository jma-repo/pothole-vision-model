"""Build review.html for detect_tesla.py results: label each detection with the keyboard and check it on a map.

    python build_review.py <results dir> --clips <folder with the source clips>

Open <results dir>/review.html in Chrome or Edge. Labels are kept in the browser as you go; "Export labels" saves them
to pothole_labels.json for retraining. Coordinates are the car's position when the pothole was last in view, so
the pothole itself is a few metres ahead along the heading.
"""
import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

NEAR_M = 20  # other detections this close may be the same pothole seen on another pass


@dataclass(frozen=True)
class Candidate:
    id: str
    clip: str
    track_id: int
    seen_at: str
    first_s: float
    last_s: float
    hits: int
    conf: float
    lat: float | None
    lon: float | None
    heading: float | None
    speed_kmh: float | None
    box: tuple[int, int, int, int]
    snapshot: str
    video: str


def _opt(value: str) -> float | None:
    return float(value) if value else None


def video_url(video: Path, page_dir: Path) -> str:
    """Relative URL from the page, so the folders can move together; absolute when they sit on different drives."""
    try:
        return quote(Path(os.path.relpath(video, page_dir)).as_posix())
    except ValueError:
        return video.resolve().as_uri()


def read_candidates(results: Path, clips: Path) -> list[Candidate]:
    videos = {p.name: p for p in clips.rglob("*.mp4")}
    out = []
    with (results / "potholes.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["clip"] not in videos:
                raise SystemExit(f"{row['clip']} is not under {clips}")
            out.append(Candidate(
                id=f"{row['clip']}#{row['track_id']}",
                clip=row["clip"],
                track_id=int(row["track_id"]),
                seen_at=row["seen_at"],
                first_s=float(row["first_s"]),
                last_s=float(row["last_s"]),
                hits=int(row["hits"]),
                conf=float(row["max_conf"]),
                lat=_opt(row["lat"]),
                lon=_opt(row["lon"]),
                heading=_opt(row["heading_deg"]),
                speed_kmh=_opt(row["speed_kmh"]),
                box=(int(row["box_x1"]), int(row["box_y1"]), int(row["box_x2"]), int(row["box_y2"])),
                snapshot=row["snapshot"],
                video=video_url(videos[row["clip"]], results),
            ))
    return out


def distance_m(a: Candidate, b: Candidate) -> float:
    p1, p2, dl = math.radians(a.lat), math.radians(b.lat), math.radians(b.lon - a.lon)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def nearby(cands: list[Candidate]) -> dict[str, list[str]]:
    located = [c for c in cands if c.lat is not None]
    return {c.id: [o.id for o in located if o is not c and distance_m(c, o) < NEAR_M] for c in located}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="the --out folder of detect_tesla.py")
    ap.add_argument("--clips", type=Path, required=True, help="folder holding the clips named in potholes.csv")
    args = ap.parse_args()

    cands = read_candidates(args.results, args.clips)
    near = nearby(cands)
    data = [asdict(c) | {"near": near.get(c.id, [])} for c in cands]
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = args.results / "review.html"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    tmp.replace(out)
    print(f"{len(cands)} detections -> {out}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pothole review</title>
<style>
:root {
  --bg: #f4f5f7; --panel: #fff; --text: #1c2230; --muted: #6b7280; --line: #e2e5ea; --hover: #eef2fb;
  --yes: #1d9a55; --no: #d04444; --unsure: #d6951a; --accent: #2f6fde;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #13161b; --panel: #1c2027; --text: #e6e8ed; --muted: #98a1b0; --line: #2b313b; --hover: #252c38; }
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 system-ui, "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); }
header { position: sticky; top: 0; z-index: 2; display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center;
  padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--line); }
h1 { font-size: 16px; margin: 0; }
.progress { flex: 1; min-width: 240px; }
#counts { font-size: 13px; color: var(--muted); }
.bar { display: flex; height: 6px; margin-top: 4px; border-radius: 3px; overflow: hidden; background: var(--line); }
.bar span { height: 100%; }
select, button, .btn { font: inherit; color: inherit; background: var(--panel); border: 1px solid var(--line);
  border-radius: 6px; padding: 4px 10px; cursor: pointer; }
button:hover, .btn:hover { background: var(--hover); }
#warn { display: none; width: 100%; color: var(--no); font-size: 13px; }
main { display: grid; grid-template-columns: 210px minmax(0, 1fr) 380px; gap: 12px; padding: 12px 16px; }
#list { display: flex; flex-direction: column; gap: 2px; max-height: calc(100vh - 84px); overflow-y: auto; }
#list button { display: flex; gap: 8px; align-items: center; text-align: left; border: 0; padding: 4px 8px;
  background: transparent; font-variant-numeric: tabular-nums; }
#list button.cur { background: var(--hover); outline: 1px solid var(--accent); }
.dot { width: 9px; height: 9px; border-radius: 50%; flex: none; border: 1px solid var(--muted); }
.dot.yes { background: var(--yes); border-color: var(--yes); }
.dot.no { background: var(--no); border-color: var(--no); }
.dot.unsure { background: var(--unsure); border-color: var(--unsure); }
#list .c { margin-left: auto; color: var(--muted); font-size: 12px; }
#viewer { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
#crop { width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 8px; }
.media { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.media img, .media video { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #000; border-radius: 8px; }
.cap { font-size: 12px; color: var(--muted); margin-top: 2px; }
aside { display: flex; flex-direction: column; gap: 10px; }
#meta { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
#meta dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; margin: 6px 0 0; }
#meta dt { color: var(--muted); }
#meta dd { margin: 0; font-variant-numeric: tabular-nums; }
.decide { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.decide button { padding: 10px 4px; font-weight: 600; color: #fff; border: 0; }
.decide .yes { background: var(--yes); } .decide .no { background: var(--no); } .decide .unsure { background: var(--unsure); }
.decide button.on { box-shadow: 0 0 0 3px var(--panel), 0 0 0 5px var(--text); }
.decide button:hover { filter: brightness(1.08); }
kbd { font: 11px ui-monospace, monospace; padding: 0 4px; border-radius: 3px; background: rgba(255,255,255,.25); }
.nav { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
#map { width: 100%; height: 300px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
.links a { color: var(--accent); }
.note, .keys { margin: 0; font-size: 12px; color: var(--muted); }
#near button { margin: 4px 6px 0 0; font-size: 12px; }
@media (max-width: 1100px) { main { grid-template-columns: 1fr; } #list { max-height: 180px; } }
</style>
</head>
<body>
<header>
  <h1>Pothole review</h1>
  <div class="progress"><div id="counts"></div><div class="bar" id="bar"></div></div>
  <label>Show <select id="filter">
    <option value="all">All</option><option value="todo">Not reviewed</option><option value="yes">Pothole</option>
    <option value="no">Not a pothole</option><option value="unsure">Unsure</option></select></label>
  <label>Sort <select id="sort"><option value="time">By time</option><option value="conf">By confidence</option></select></label>
  <button id="export">Export labels</button>
  <label class="btn">Import<input type="file" id="import" accept=".json" hidden></label>
  <div id="warn">This browser won't save labels locally. Export them before you close the page.</div>
</header>
<main>
  <nav id="list"></nav>
  <section id="viewer">
    <canvas id="crop" width="1280" height="720"></canvas>
    <div class="media">
      <div><img id="frame" alt="Full frame"><div class="cap">Full frame (red box: the detection)</div></div>
      <div><video id="video" muted playsinline controls></video><div class="cap">The seconds around it, looped at 0.5× speed</div></div>
    </div>
  </section>
  <aside>
    <div id="meta"></div>
    <div class="decide">
      <button class="yes" data-label="yes">Pothole <kbd>Y</kbd></button>
      <button class="no" data-label="no">Not a pothole <kbd>N</kbd></button>
      <button class="unsure" data-label="unsure">Unsure <kbd>U</kbd></button>
    </div>
    <div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button></div>
    <iframe id="map" title="Satellite map"></iframe>
    <div class="links"><a id="gmaps" target="_blank" rel="noopener">Open in Google Maps</a> ·
      <a id="street" target="_blank" rel="noopener">Street View (facing the direction of travel)</a></div>
    <p class="note">The pin is the car's position when the pothole was last in view; the pothole is a few metres ahead.</p>
    <div id="near"></div>
    <p class="keys">Keys: Y pothole · N not a pothole · U unsure · Backspace clear · ← → move · Space play/pause</p>
  </aside>
</main>
<script>
const DATA = __DATA__;
const KEY = "pothole-review-labels";
const NAMES = { yes: "pothole", no: "not a pothole", unsure: "unsure" };
const COLORS = { yes: "var(--yes)", no: "var(--no)", unsure: "var(--unsure)" };
const $ = id => document.getElementById(id);
const byId = new Map(DATA.map((c, i) => [c.id, i]));
let labels = loadLabels();
let order = [];
let pos = 0;
let seg = null;

function loadLabels() {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
}
function saveLabels() {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); $("warn").style.display = "none"; }
  catch { $("warn").style.display = "block"; }
}
const labelOf = c => labels[c.id]?.label;
const shortTime = c => c.seen_at.slice(5, 19);

function rebuild(keepIndex) {
  const f = $("filter").value;
  order = DATA.map((_, i) => i).filter(i => {
    const l = labelOf(DATA[i]);
    return f === "all" || (f === "todo" ? !l : l === f);
  });
  if ($("sort").value === "conf") order.sort((a, b) => DATA[b].conf - DATA[a].conf);
  else order.sort((a, b) => DATA[a].seen_at.localeCompare(DATA[b].seen_at));
  const list = $("list");
  list.replaceChildren(...order.map((i, p) => {
    const c = DATA[i], b = document.createElement("button");
    const dot = document.createElement("span"), t = document.createElement("span"), conf = document.createElement("span");
    dot.className = "dot " + (labelOf(c) || "");
    t.textContent = shortTime(c);
    conf.className = "c"; conf.textContent = c.conf.toFixed(2);
    b.append(dot, t, conf);
    b.onclick = () => { pos = p; show(); };
    return b;
  }));
  const at = order.indexOf(keepIndex);
  pos = at >= 0 ? at : 0;
  show();
}

function counts() {
  const n = { yes: 0, no: 0, unsure: 0 };
  for (const c of DATA) { const l = labelOf(c); if (l) n[l]++; }
  const done = n.yes + n.no + n.unsure;
  $("counts").textContent = `Reviewed ${done} / ${DATA.length} · pothole ${n.yes} · not a pothole ${n.no} · unsure ${n.unsure}` +
    (done === DATA.length ? " · all done, remember to export" : "");
  $("bar").replaceChildren(...["yes", "no", "unsure"].map(k => {
    const s = document.createElement("span");
    s.style.width = (100 * n[k] / DATA.length) + "%"; s.style.background = COLORS[k];
    return s;
  }));
}

function show() {
  counts();
  if (!order.length) { $("meta").textContent = "No detections match this filter."; return; }
  const c = DATA[order[pos]];
  [...$("list").children].forEach((b, p) => b.classList.toggle("cur", p === pos));
  $("list").children[pos].scrollIntoView({ block: "nearest" });
  const rows = [["Time", c.seen_at.slice(0, 19)], ["Clip", c.clip], ["Confidence", c.conf.toFixed(2)],
    ["Frames detected", c.hits], ["Speed", c.speed_kmh == null ? "—" : c.speed_kmh + " km/h"],
    ["Heading", c.heading == null ? "—" : Math.round(c.heading) + "°"]];
  const head = document.createElement("div");
  head.textContent = `${pos + 1} of ${order.length}` + (labelOf(c) ? ` · labelled ${NAMES[labelOf(c)]}` : "");
  const dl = document.createElement("dl");
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = v; dl.append(dt, dd);
  }
  $("meta").replaceChildren(head, dl);
  document.querySelectorAll(".decide button").forEach(b => b.classList.toggle("on", b.dataset.label === labelOf(c)));

  const img = $("frame");
  img.onload = () => drawCrop(img, c.box);
  img.src = c.snapshot;

  seg = { start: Math.max(0, c.first_s - 1.5), end: c.last_s + 0.5 };
  const v = $("video");
  v.src = c.video;
  v.onloadedmetadata = () => { v.currentTime = seg.start; v.playbackRate = 0.5; v.play().catch(() => {}); };

  if (c.lat == null) {
    $("map").src = "about:blank";
    $("gmaps").removeAttribute("href"); $("street").removeAttribute("href");
  } else {
    $("map").src = `https://maps.google.com/maps?q=${c.lat},${c.lon}&t=k&z=19&output=embed`;
    $("gmaps").href = `https://www.google.com/maps/search/?api=1&query=${c.lat},${c.lon}`;
    $("street").href = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${c.lat},${c.lon}` +
      `&heading=${Math.round(c.heading || 0)}&pitch=-10&fov=80`;
  }

  const near = $("near");
  near.replaceChildren();
  if (c.near.length) {
    near.append(`${c.near.length} other detection${c.near.length > 1 ? "s" : ""} within 20 m, possibly the same pothole: `);
    for (const id of c.near) {
      const o = DATA[byId.get(id)], b = document.createElement("button");
      b.textContent = shortTime(o) + (labelOf(o) ? " · " + NAMES[labelOf(o)] : "");
      b.onclick = () => jump(id);
      near.append(b);
    }
  }
}

function drawCrop(img, [x1, y1, x2, y2]) {
  const cv = $("crop"), ctx = cv.getContext("2d"), aspect = cv.width / cv.height;
  const nw = img.naturalWidth, nh = img.naturalHeight;
  let w = Math.max((x2 - x1) * 3, (y2 - y1) * 3 * aspect, 360), h = w / aspect;
  if (w > nw) { w = nw; h = w / aspect; }
  if (h > nh) { h = nh; w = h * aspect; }
  const sx = Math.min(Math.max((x1 + x2 - w) / 2, 0), nw - w), sy = Math.min(Math.max((y1 + y2 - h) / 2, 0), nh - h);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, sx, sy, w, h, 0, 0, cv.width, cv.height);
}

function jump(id) {
  const i = byId.get(id);
  if (!order.includes(i)) { $("filter").value = "all"; }
  rebuild(i);
}

function mark(label) {
  if (!order.length) return;
  const c = DATA[order[pos]];
  if (label) labels[c.id] = { label, at: new Date().toISOString() }; else delete labels[c.id];
  saveLabels();
  const dot = $("list").children[pos].querySelector(".dot");
  dot.className = "dot " + (label || "");
  if (!label) { show(); return; }
  const next = order.findIndex((i, p) => p > pos && !labelOf(DATA[i]));
  pos = next >= 0 ? next : Math.min(pos + 1, order.length - 1);
  show();
}

function go(step) {
  if (!order.length) return;
  pos = Math.min(Math.max(pos + step, 0), order.length - 1);
  show();
}

function exportLabels() {
  const rows = DATA.filter(labelOf).map(c => ({
    id: c.id, clip: c.clip, track_id: c.track_id, label: labelOf(c), labeled_at: labels[c.id].at,
    seen_at: c.seen_at, lat: c.lat, lon: c.lon, conf: c.conf, box: c.box, snapshot: c.snapshot,
  }));
  const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), total: DATA.length, labels: rows }, null, 1)],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "pothole_labels.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function importLabels(file) {
  const parsed = JSON.parse(await file.text());
  let n = 0;
  for (const r of parsed.labels || []) {
    if (byId.has(r.id) && NAMES[r.label]) { labels[r.id] = { label: r.label, at: r.labeled_at }; n++; }
  }
  saveLabels();
  rebuild(order[pos]);
  alert(`Imported ${n} labels`);
}

$("video").addEventListener("timeupdate", () => {
  const v = $("video");
  if (seg && v.currentTime > seg.end) v.currentTime = seg.start;
});
document.addEventListener("click", e => e.target.closest("button")?.blur());  // so Space can't re-press it
document.querySelectorAll(".decide button").forEach(b => b.onclick = () => mark(b.dataset.label));
$("prev").onclick = () => go(-1);
$("next").onclick = () => go(1);
$("filter").onchange = () => rebuild(order[pos]);
$("sort").onchange = () => rebuild(order[pos]);
$("export").onclick = exportLabels;
$("import").onchange = e => { if (e.target.files[0]) importLabels(e.target.files[0]); e.target.value = ""; };
document.addEventListener("keydown", e => {
  if (e.target.closest("select, input") || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "y" || k === "1") mark("yes");
  else if (k === "n" || k === "2") mark("no");
  else if (k === "u" || k === "3") mark("unsure");
  else if (e.key === "Backspace") mark(null);
  else if (e.key === "ArrowRight" || e.key === "ArrowDown") go(1);
  else if (e.key === "ArrowLeft" || e.key === "ArrowUp") go(-1);
  else if (e.key === " ") { const v = $("video"); v.paused ? v.play() : v.pause(); }
  else return;
  e.preventDefault();
});
rebuild(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
