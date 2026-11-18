"""Build map.html: reviewed potholes on a city map, ranked by rough estimates of what they cost drivers and the city.

    python build_map.py <results dir> [--labels pothole_labels.json]

Review labels come from the exported JSON, or, without --labels, from the browser: open map.html in the browser you
reviewed in and it reads them from there. All dollar figures are rough, CAD, and come from the assumptions below,
which the page also lists.
"""
import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

# Tesla HW3 main forward camera as recorded at 1280x960: 50° horizontal field of view, about 1.4 m above the road.
IMAGE_W, IMAGE_H = 1280, 960
FOCAL_PX = (IMAGE_W / 2) / math.tan(math.radians(25))
CAMERA_HEIGHT_M = 1.4
HORIZON_ROW = 470  # read off straight, level roads; slopes move it, so far-away sizes are least reliable
LAST_SEEN_AHEAD_M = 4.5  # a pothole drops out of the bottom of the frame about this far ahead of the camera
HALF_TRACK_M = 0.8  # each wheel runs about this far left or right of the camera

# Repair: Calgary's $6.9M pothole program filled about 36,000 potholes in 2024 (about $190 each);
# this base plus area rate puts a 0.3 m² pothole at that average.
REPAIR_BASE, REPAIR_PER_M2 = 120, 250
# Damage: CAA puts a pothole repair bill at roughly $300-$900 (tires, rims, alignment, suspension).
DAMAGE_PER_HIT = 600
# Traffic in the pothole's lane, by road type guessed from the car's speed.
LANE_VEHICLES_PER_DAY = {"local": 300, "collector": 2_500, "arterial": 7_500, "freeway": 15_000}
# Share of passing vehicles that put a wheel in it: in a wheel path, or elsewhere in the lane.
P_HIT = {True: 0.25, False: 0.03}
# Chance a hit needs repairs, for a 0.3 m² pothole at 50 km/h. Chosen so a typical arterial wheel-path pothole
# damages about one car a week; scaled by size and speed.
P_DAMAGE = 6e-5
EXPOSURE_DAYS = 30


@dataclass(frozen=True)
class Estimate:
    id: str
    clip: str
    seen_at: str
    conf: float
    lat: float
    lon: float
    heading: float
    speed_kmh: float
    road: str
    width_m: float
    length_m: float
    area_m2: float
    wheel_path: bool
    repair_cad: float
    damage_cad: float
    snapshot: str
    box: tuple[int, int, int, int]


def ground_distance(row: float) -> float:
    """Metres from the camera to the road at an image row, for a flat road."""
    return CAMERA_HEIGHT_M * FOCAL_PX / max(row - HORIZON_ROW, 20)


def road_type(speed_kmh: float) -> str:
    if speed_kmh < 30:
        return "local"
    if speed_kmh < 55:
        return "collector"
    if speed_kmh < 80:
        return "arterial"
    return "freeway"


def move(lat: float, lon: float, heading_deg: float, ahead_m: float, right_m: float) -> tuple[float, float]:
    h = math.radians(heading_deg)
    east = ahead_m * math.sin(h) + right_m * math.cos(h)
    north = ahead_m * math.cos(h) - right_m * math.sin(h)
    return lat + north / 111_320, lon + east / (111_320 * math.cos(math.radians(lat)))


def estimate(row: dict[str, str]) -> Estimate:
    x1, y1, x2, y2 = (int(row[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2"))
    d_mid = ground_distance((y1 + y2) / 2)
    width = (x2 - x1) * d_mid / FOCAL_PX
    # Width scales linearly with distance, but at the 6-10 m most snapshots are taken a few pixels of box height are
    # metres of road (measured lengths came out ~4x the widths), so length is held to 1.5x the width.
    length = min(max(ground_distance(y1) - ground_distance(y2), 0.05), 1.5 * width)
    area = min(math.pi / 4 * width * length, 4.0)
    lateral = ((x1 + x2) / 2 - IMAGE_W / 2) * d_mid / FOCAL_PX
    wheel_path = any(abs(lateral - side * HALF_TRACK_M) < width / 2 + 0.1 for side in (-1, 1))
    speed = float(row["speed_kmh"])
    road = road_type(speed)
    size_factor = min(max(area / 0.3, 0.3), 3.0)
    speed_factor = min(max(speed / 50, 0.5), 2.0)
    hits_per_day = LANE_VEHICLES_PER_DAY[road] * P_HIT[wheel_path] * P_DAMAGE * size_factor * speed_factor
    heading = float(row["heading_deg"])
    lat, lon = move(float(row["lat"]), float(row["lon"]), heading, LAST_SEEN_AHEAD_M, lateral)
    return Estimate(
        id=f"{row['clip']}#{row['track_id']}", clip=row["clip"], seen_at=row["seen_at"][:19],
        conf=float(row["max_conf"]), lat=round(lat, 7), lon=round(lon, 7), heading=heading, speed_kmh=speed,
        road=road, width_m=round(width, 2), length_m=round(length, 2), area_m2=round(area, 2),
        wheel_path=wheel_path, repair_cad=round(REPAIR_BASE + REPAIR_PER_M2 * area),
        damage_cad=round(hits_per_day * EXPOSURE_DAYS * DAMAGE_PER_HIT), snapshot=row["snapshot"],
        box=(x1, y1, x2, y2),
    )


def read_labels(path: Path) -> dict[str, str]:
    return {r["id"]: r["label"] for r in json.loads(path.read_text(encoding="utf-8"))["labels"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="the --out folder of detect_tesla.py")
    ap.add_argument("--labels", type=Path, help="pothole_labels.json exported from review.html")
    args = ap.parse_args()

    with (args.results / "potholes.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    located = [r for r in rows if r["lat"]]
    data = [asdict(estimate(r)) for r in located]
    labels = read_labels(args.labels) if args.labels else {}
    params = {
        "repair": f"${REPAIR_BASE} + ${REPAIR_PER_M2} per m² (Calgary averages about $190 per pothole: "
                  "a $6.9M program, about 36,000 potholes filled in 2024)",
        "damage": f"${DAMAGE_PER_HIT} per damaging hit (CAA: roughly $300-$900)",
        "traffic": ", ".join(f"{k} {v:,}" for k, v in LANE_VEHICLES_PER_DAY.items()) + " vehicles/day in the lane",
        "hit": f"{P_HIT[True]:.0%} of vehicles in a wheel path, {P_HIT[False]:.0%} elsewhere",
        "p_damage": f"{P_DAMAGE * 1e5:g} in 100,000 hits needs repairs (0.3 m² at 50 km/h), scaled by size and speed",
        "days": EXPOSURE_DAYS,
    }
    html = (TEMPLATE.replace("__DATA__", json.dumps(data))
            .replace("__LABELS__", json.dumps(labels))
            .replace("__PARAMS__", json.dumps(params, ensure_ascii=False)))
    out = args.results / "map.html"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(out)
    print(f"{len(data)} located detections ({len(rows) - len(located)} without GPS left out), "
          f"{len(labels)} labels embedded -> {out}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pothole map</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root { --bg: #fff; --text: #1c2230; --muted: #6b7280; --line: #e2e5ea; --hover: #eef2fb; --accent: #2f6fde; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1c2027; --text: #e6e8ed; --muted: #98a1b0; --line: #2b313b; --hover: #252c38; }
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { display: grid; grid-template-columns: 380px 1fr; font: 14px/1.45 system-ui, sans-serif; background: var(--bg); color: var(--text); }
aside { display: flex; flex-direction: column; min-height: 0; border-right: 1px solid var(--line); }
.top { padding: 14px 16px 10px; border-bottom: 1px solid var(--line); }
h1 { font-size: 17px; margin: 0 0 6px; }
#summary { color: var(--muted); font-size: 13px; }
#summary b { color: var(--text); }
.controls { display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 10px; font-size: 13px; }
select, button { font: inherit; color: inherit; background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 3px 8px; }
#notice { display: none; margin-top: 8px; padding: 8px 10px; border-radius: 6px; background: #fff4d6; color: #6b4e00; font-size: 13px; }
#list { flex: 1; overflow-y: auto; }
#list button { display: grid; grid-template-columns: 34px 1fr auto; gap: 2px 8px; width: 100%; text-align: left;
  border: 0; border-bottom: 1px solid var(--line); border-radius: 0; padding: 8px 16px; cursor: pointer; }
#list button:hover, #list button.cur { background: var(--hover); }
.rank { color: var(--muted); font-variant-numeric: tabular-nums; }
.val { font-weight: 600; font-variant-numeric: tabular-nums; }
.sub { grid-column: 2 / 4; color: var(--muted); font-size: 12px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
details { padding: 10px 16px; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); }
details summary { cursor: pointer; color: var(--text); }
details li { margin: 3px 0; }
#map { height: 100%; }
.legend { background: var(--bg); color: var(--text); padding: 8px 10px; border-radius: 6px; font-size: 12px; line-height: 1.7; box-shadow: 0 1px 4px rgba(0,0,0,.3); }
.pop { width: 260px; font-size: 12.5px; }
.pop .crop { width: 260px; height: 195px; border-radius: 6px; background-repeat: no-repeat; background-color: #000; }
.pop table { width: 100%; margin-top: 6px; border-collapse: collapse; }
.pop td { padding: 1px 0; } .pop td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.pop a { color: var(--accent); }
@media (max-width: 800px) { body { grid-template-columns: 1fr; grid-template-rows: 45vh 55vh; } aside { order: 2; } }
</style>
</head>
<body>
<aside>
  <div class="top">
    <h1>Pothole map</h1>
    <div id="summary"></div>
    <div class="controls">
      <label>Colour by <select id="metric">
        <option value="damage_cad">Driver damage, next 30 days</option>
        <option value="repair_cad">Repair cost</option>
        <option value="ratio">Damage avoided per repair dollar</option></select></label>
      <label><input type="checkbox" id="unsure"> include "unsure"</label>
      <label><button id="importBtn">Import labels</button><input type="file" id="import" accept=".json" hidden></label>
    </div>
    <div id="notice"></div>
  </div>
  <nav id="list"></nav>
  <details>
    <summary>How the dollar figures are estimated</summary>
    <ul id="params"></ul>
    <p>Size comes from the detection box and the camera's geometry, assuming a flat road. Road type is guessed from the
      car's speed. Repeat sightings within 15 m are merged into one pothole. The numbers are rough; use them to rank
      potholes against each other rather than as quotes.</p>
  </details>
</aside>
<div id="map"></div>
<script>
const DATA = __DATA__;
const EMBEDDED = __LABELS__;
const PARAMS = __PARAMS__;
const MERGE_M = 15;
const METRICS = {
  damage_cad: { name: "Driver damage, next 30 days", fmt: v => "$" + Math.round(v).toLocaleString(), bins: [100, 500, 2000] },
  repair_cad: { name: "Repair cost", fmt: v => "$" + Math.round(v).toLocaleString(), bins: [150, 250, 400] },
  ratio: { name: "Damage avoided per repair dollar", fmt: v => v.toFixed(1) + "×", bins: [0.5, 2, 10] },
};
const COLORS = ["#2e9e5b", "#e0b324", "#e8702a", "#d23b3b"];
const $ = id => document.getElementById(id);

function storedLabels() {
  try {
    const raw = JSON.parse(localStorage.getItem("pothole-review-labels")) || {};
    return Object.fromEntries(Object.entries(raw).map(([id, v]) => [id, v.label]));
  } catch { return {}; }
}
let labels = Object.keys(EMBEDDED).length ? EMBEDDED : storedLabels();

function dist(a, b) {
  const dy = (a.lat - b.lat) * 111320, dx = (a.lon - b.lon) * 111320 * Math.cos(a.lat * Math.PI / 180);
  return Math.hypot(dx, dy);
}

function potholes() {
  const reviewed = Object.keys(labels).length > 0;
  const keep = DATA.filter(d => !reviewed || labels[d.id] === "yes" || ($("unsure").checked && labels[d.id] === "unsure"));
  let groups = [];
  for (const d of keep) {
    const near = groups.filter(g => g.some(o => dist(o, d) < MERGE_M));
    groups = groups.filter(g => !near.includes(g));
    groups.push([...near.flat(), d]);
  }
  return groups.map(g => {
    const top = g.reduce((a, b) => (b.damage_cad > a.damage_cad ? b : a));
    const times = g.map(x => x.seen_at).sort();
    return { ...top, ratio: top.damage_cad / top.repair_cad, sightings: g.length, unsure: g.every(x => labels[x.id] === "unsure"),
             first: times[0], last: times[times.length - 1] };
  });
}

const map = L.map("map");
const streets = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Tiles © Esri" }).addTo(map);
const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Tiles © Esri" });
L.control.layers({ Streets: streets, Satellite: satellite }).addTo(map);
const layer = L.layerGroup().addTo(map);
const legend = L.control({ position: "bottomright" });
legend.onAdd = () => L.DomUtil.create("div", "legend");
legend.addTo(map);
let markers = [];

function colorOf(metric, v) {
  const bins = METRICS[metric].bins;
  return COLORS[bins.filter(b => v >= b).length];
}

function popup(p) {
  const el = document.createElement("div");
  el.className = "pop";
  const [x1, y1, x2, y2] = p.box, aspect = 4 / 3;
  let w = Math.max((x2 - x1) * 3, (y2 - y1) * 3 * aspect, 320), h = w / aspect;
  if (h > 960) { h = 960; w = h * aspect; }
  const sx = Math.min(Math.max((x1 + x2 - w) / 2, 0), 1280 - w), sy = Math.min(Math.max((y1 + y2 - h) / 2, 0), 960 - h);
  const scale = 260 / w;
  const crop = document.createElement("div");
  crop.className = "crop";
  crop.style.backgroundImage = `url("${p.snapshot}")`;
  crop.style.backgroundSize = `${1280 * scale}px ${960 * scale}px`;
  crop.style.backgroundPosition = `${-sx * scale}px ${-sy * scale}px`;
  const rows = [
    ["Driver damage, next 30 days", METRICS.damage_cad.fmt(p.damage_cad)],
    ["Repair cost", METRICS.repair_cad.fmt(p.repair_cad)],
    ["Estimated size", `${p.width_m} × ${p.length_m} m (${p.area_m2} m²)`],
    ["In a wheel path", p.wheel_path ? "yes" : "no"],
    ["Road type (from speed)", `${p.road}, ${Math.round(p.speed_kmh)} km/h`],
    ["Sightings", p.sightings],
    ["Seen", p.first === p.last ? p.first : `${p.first.slice(0, 10)} to ${p.last.slice(0, 10)}`],
  ];
  const table = document.createElement("table");
  for (const [k, v] of rows) {
    const tr = table.insertRow();
    tr.insertCell().textContent = k;
    tr.insertCell().textContent = v;
  }
  const links = document.createElement("div");
  links.style.marginTop = "6px";
  const sv = document.createElement("a"), gm = document.createElement("a");
  sv.href = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${p.lat},${p.lon}&heading=${Math.round(p.heading)}&pitch=-10&fov=80`;
  gm.href = `https://www.google.com/maps/search/?api=1&query=${p.lat},${p.lon}`;
  sv.textContent = "Street View"; gm.textContent = "Google Maps";
  sv.target = gm.target = "_blank";
  links.append(sv, " · ", gm);
  el.append(crop, table, links);
  return el;
}

function render(fit) {
  const metric = $("metric").value, m = METRICS[metric];
  const list = potholes().sort((a, b) => b[metric] - a[metric]);
  layer.clearLayers();
  markers = list.map(p => L.circleMarker([p.lat, p.lon], {
    radius: 7 + Math.min(Math.sqrt(p.area_m2) * 6, 8), color: "#222", weight: 1, dashArray: p.unsure ? "3" : null,
    fillColor: colorOf(metric, p[metric]), fillOpacity: 0.9,
  }).bindPopup(() => popup(p), { maxWidth: 280 }).addTo(layer));
  $("list").replaceChildren(...list.map((p, i) => {
    const b = document.createElement("button");
    const rank = document.createElement("span"), road = document.createElement("span"), val = document.createElement("span"), sub = document.createElement("span");
    rank.className = "rank"; rank.textContent = "#" + (i + 1);
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = colorOf(metric, p[metric]);
    road.append(sw, `${p.road} road${p.wheel_path ? ", wheel path" : ""}`);
    val.className = "val"; val.textContent = m.fmt(p[metric]);
    sub.className = "sub";
    sub.textContent = `${p.area_m2} m² · repair ${METRICS.repair_cad.fmt(p.repair_cad)} · ${p.sightings} sighting${p.sightings > 1 ? "s" : ""} · ${p.last.slice(0, 10)}`;
    b.append(rank, road, val, sub);
    b.onclick = () => {
      document.querySelectorAll("#list button").forEach(x => x.classList.toggle("cur", x === b));
      map.flyTo([p.lat, p.lon], 18, { duration: 0.6 });
      markers[i].openPopup();
    };
    return b;
  }));
  const total = k => list.reduce((s, p) => s + p[k], 0);
  const reviewed = Object.keys(labels).length > 0;
  $("summary").innerHTML = "";
  $("summary").append(
    reviewed ? `${list.length} potholes from ${DATA.filter(d => labels[d.id] === "yes").length} confirmed detections. ` :
               `${list.length} unreviewed detections. `,
    "Repair all: ", Object.assign(document.createElement("b"), { textContent: METRICS.repair_cad.fmt(total("repair_cad")) }),
    ". Driver damage if left 30 days: ", Object.assign(document.createElement("b"), { textContent: METRICS.damage_cad.fmt(total("damage_cad")) }), ".");
  $("notice").style.display = reviewed ? "none" : "block";
  $("notice").textContent = "No review labels found, so every detection is shown. Open this page in the browser you " +
    "reviewed in, or import pothole_labels.json.";
  const legendEl = document.querySelector(".legend");
  const b = m.bins;
  legendEl.innerHTML = "";
  legendEl.append(Object.assign(document.createElement("b"), { textContent: m.name }));
  [`< ${m.fmt(b[0])}`, `${m.fmt(b[0])} – ${m.fmt(b[1])}`, `${m.fmt(b[1])} – ${m.fmt(b[2])}`, `≥ ${m.fmt(b[2])}`].forEach((t, i) => {
    const row = document.createElement("div");
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = COLORS[i];
    row.append(sw, t);
    legendEl.append(row);
  });
  if (fit && list.length) map.fitBounds(L.latLngBounds(list.map(p => [p.lat, p.lon])).pad(0.1));
}

$("params").replaceChildren(...[
  ["Repair", PARAMS.repair], ["Damage", PARAMS.damage], ["Traffic", PARAMS.traffic],
  ["Hits", PARAMS.hit], ["Damage per hit", PARAMS.p_damage], ["Horizon", `${PARAMS.days} days`],
].map(([k, v]) => { const li = document.createElement("li"); li.textContent = `${k}: ${v}`; return li; }));
$("metric").onchange = () => render(false);
$("unsure").onchange = () => render(false);
$("importBtn").onclick = () => $("import").click();
$("import").onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  labels = Object.fromEntries((JSON.parse(await f.text()).labels || []).map(r => [r.id, r.label]));
  e.target.value = "";
  render(true);
};
if (DATA.length) render(true); else map.setView([51.045, -114.06], 11);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
