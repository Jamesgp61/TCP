#!/usr/bin/env python3
"""
dashboard.py
============
Visual tracking interface for the open-source 6-bit Topological
Energy-Based Model (TEBM) framework.

Reads ``topology_map.json`` from the workspace directory, parses the
current 6-bit node-to-file distribution and active edge weights, and
emits a **completely self-contained** ``topology.html`` file containing
an interactive 8×8 visual matrix (64 nodes) rendered with raw inline
SVG + CSS + vanilla JavaScript.

Zero external dependencies:
  • No third-party Python packages
  • No JavaScript CDNs
  • No web fonts loaded from the network
  • Pure Python string compilation → static HTML

Expected ``topology_map.json`` schema
-------------------------------------
{
  "nodes": {
    "0": {
      "bitstring": "000000",
      "files": ["/abs/path/to/file_a.py", "/abs/path/to/file_b.py"],
      "energy": 0.42
    },
    ...
    "63": { ... }
  },
  "edges": [
    {"source": 0, "target": 1, "weight": 0.75},
    ...
  ]
}

If the file is missing, a synthetic demo topology is generated so the
dashboard can be previewed immediately.

Usage
-----
    python dashboard.py
    # → writes /workspace/topology_workspace/topology.html
"""

from __future__ import annotations

import json
import os
import html
import random
from datetime import datetime
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKSPACE_DIR = "/workspace/topology_workspace"
TOPOLOGY_MAP_PATH = os.path.join(WORKSPACE_DIR, "topology_map.json")
OUTPUT_HTML_PATH = os.path.join(WORKSPACE_DIR, "topology.html")

GRID_N = 8          # 8×8 matrix
TOTAL_NODES = 64    # 6-bit → 2^6 nodes
BITS = 6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_topology_map(path: str = TOPOLOGY_MAP_PATH) -> Dict[str, Any]:
    """Load and normalise ``topology_map.json``.

    Falls back to a synthetic demo topology when the file is absent so
    the dashboard remains usable during development.
    """
    if not os.path.exists(path):
        print(f"[dashboard] '{path}' not found — generating synthetic demo topology.")
        return _synthetic_topology()

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    raw_nodes = raw.get("nodes", {})
    raw_edges = raw.get("edges", [])

    nodes: Dict[int, Dict[str, Any]] = {}
    for i in range(TOTAL_NODES):
        entry = raw_nodes.get(str(i), {})
        bitstring = entry.get("bitstring", format(i, f"0{BITS}b"))
        files = entry.get("files", []) or []
        energy = float(entry.get("energy", 0.0))
        nodes[i] = {
            "index": i,
            "bitstring": bitstring,
            "files": [str(f) for f in files],
            "energy": energy,
            "density": len(files),
        }

    edges: List[Dict[str, Any]] = []
    for e in raw_edges:
        edges.append({
            "source": int(e.get("source", 0)),
            "target": int(e.get("target", 0)),
            "weight": float(e.get("weight", 0.0)),
        })

    return {"nodes": nodes, "edges": edges}


def _synthetic_topology() -> Dict[str, Any]:
    """Produce a deterministic demo topology (Hamming-1 edges)."""
    rng = random.Random(42)
    nodes: Dict[int, Dict[str, Any]] = {}
    for i in range(TOTAL_NODES):
        bs = format(i, f"0{BITS}b")
        n_files = rng.randint(0, 9)
        files = [f"/workspace/topology_workspace/nodes/{bs}/module_{j}.py"
                 for j in range(n_files)]
        nodes[i] = {
            "index": i,
            "bitstring": bs,
            "files": files,
            "energy": round(rng.uniform(0, 1), 4),
            "density": n_files,
        }

    edges: List[Dict[str, Any]] = []
    for i in range(TOTAL_NODES):
        for bit in range(BITS):
            neighbour = i ^ (1 << bit)
            if neighbour > i:
                edges.append({
                    "source": i,
                    "target": neighbour,
                    "weight": round(rng.uniform(0.05, 1.0), 4),
                })

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _density_colour(density: int, max_density: int) -> str:
    """Map file density to an RGB string on a deep-blue → crimson ramp."""
    if max_density <= 0:
        return "#161628"
    t = min(density / max_density, 1.0)
    # Piecewise interpolation through three stops
    if t < 0.5:
        u = t / 0.5
        r = round(22 + (60 - 22) * u)
        g = round(28 + (120 - 28) * u)
        b = round(72 + (200 - 72) * u)
    else:
        u = (t - 0.5) / 0.5
        r = round(60 + (220 - 60) * u)
        g = round(120 + (60 - 120) * u)
        b = round(200 + (50 - 200) * u)
    return f"rgb({r},{g},{b})"


def _energy_to_bar(energy: float, max_energy: float) -> str:
    """Return a percentage string for the energy bar width."""
    if max_energy <= 0:
        return "0%"
    return f"{min(energy / max_energy, 1.0) * 100:.1f}%"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def generate_topology_html(
    topology_data: Dict[str, Any],
    output_path: str = OUTPUT_HTML_PATH,
) -> str:
    """Render the standalone ``topology.html`` file and return its path."""
    nodes: Dict[int, Dict[str, Any]] = topology_data["nodes"]
    edges: List[Dict[str, Any]] = topology_data["edges"]

    max_density = max((n["density"] for n in nodes.values()), default=1) or 1
    max_energy = max((n["energy"] for n in nodes.values()), default=1.0) or 1.0

    total_files = sum(n["density"] for n in nodes.values())
    total_edges = len(edges)
    avg_energy = (
        sum(n["energy"] for n in nodes.values()) / TOTAL_NODES if TOTAL_NODES else 0
    )

    # ---- SVG geometry ---------------------------------------------------
    cell = 64
    pad = 10
    step = cell + pad
    svg_w = step * GRID_N + pad
    svg_h = step * GRID_N + pad

    # ---- Edges (rendered behind nodes) ----------------------------------
    edge_lines: List[str] = []
    for e in edges:
        src = nodes.get(e["source"])
        tgt = nodes.get(e["target"])
        if not src or not tgt:
            continue
        sr, sc = divmod(src["index"], GRID_N)
        tr, tc = divmod(tgt["index"], GRID_N)
        x1 = pad + sc * step + cell / 2
        y1 = pad + sr * step + cell / 2
        x2 = pad + tc * step + cell / 2
        y2 = pad + tr * step + cell / 2
        op = 0.08 + 0.55 * min(e["weight"], 1.0)
        edge_lines.append(
            f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="rgba(120,170,255,{op:.3f})" stroke-width="1.4" />'
        )
    edges_svg = "\n".join(edge_lines)

    # ---- Nodes ----------------------------------------------------------
    node_groups: List[str] = []
    node_json: List[Dict[str, Any]] = []
    for i in range(TOTAL_NODES):
        n = nodes[i]
        r, c = divmod(i, GRID_N)
        x = pad + c * step
        y = pad + r * step
        colour = _density_colour(n["density"], max_density)
        energy_pct = _energy_to_bar(n["energy"], max_energy)

        files_escaped = (
            "".join(
                f'<div class="fp">{html.escape(f)}</div>' for f in n["files"]
            )
            if n["files"]
            else '<div class="fp empty">— no files mapped —</div>'
        )

        node_json.append({
            "index": i,
            "bitstring": n["bitstring"],
            "files": n["files"],
            "energy": n["energy"],
            "density": n["density"],
            "row": r,
            "col": c,
        })

        node_groups.append(f"""
      <g class="node" data-i="{i}" transform="translate({x:.1f},{y:.1f})">
        <rect class="cell" width="{cell}" height="{cell}" rx="7" ry="7"
              fill="{colour}" />
        <rect class="energy-bar" x="4" y="{cell - 7}" width="{cell - 8}" height="3"
              rx="1.5" fill="rgba(255,255,255,0.12)" />
        <rect class="energy-fill" x="4" y="{cell - 7}" width="{(cell - 8) * min(n['energy'] / max_energy, 1.0):.1f}" height="3"
              rx="1.5" fill="rgba(255,220,120,0.9)" />
        <text class="bs" x="{cell / 2}" y="{cell / 2 - 2}" text-anchor="middle">{n['bitstring']}</text>
        <text class="dens" x="{cell - 5}" y="13" text-anchor="end">{n['density']}</text>
      </g>""")

    nodes_svg = "\n".join(node_groups)
    nodes_json_str = json.dumps(node_json, separators=(",", ":"))

    # ---- Legend stops ---------------------------------------------------
    legend_stops = []
    for k in range(6):
        d = int(max_density * k / 5)
        legend_stops.append(
            f'<span class="stop" style="background:{_density_colour(d, max_density)}"></span>'
        )
    legend_html = "".join(legend_stops)

    # ---- Assemble HTML --------------------------------------------------
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M UTC")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>6-bit TEBM Topology Dashboard</title>
<style>
  :root {{
    --bg:        #0b0b14;
    --panel:     #14141f;
    --panel-2:   #1c1c2b;
    --text:      #e6e6f0;
    --muted:     #8a8aa3;
    --accent:    #7aa2ff;
    --accent-2:  #ffcc66;
    --border:    rgba(255,255,255,0.08);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    background: var(--bg);
    color: var(--text);
    font-family: "SF Mono", "Menlo", "Consolas", "Liberation Mono", monospace;
    font-size: 13px;
    line-height: 1.5;
    min-height: 100vh;
  }}
  .wrap {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 24px 60px;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 22px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{
    font-size: 20px;
    font-weight: 600;
    letter-spacing: 0.3px;
  }}
  header h1 .sub {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 400;
    margin-left: 8px;
  }}
  header .meta {{
    color: var(--muted);
    font-size: 11px;
    text-align: right;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 22px;
  }}
  .stat {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }}
  .stat .label {{
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }}
  .stat .value {{
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
  }}
  .stat .value.warm {{ color: var(--accent-2); }}

  .main {{
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 20px;
    align-items: start;
  }}
  @media (max-width: 920px) {{
    .main {{ grid-template-columns: 1fr; }}
    .stats {{ grid-template-columns: repeat(2, 1fr); }}
  }}

  .canvas {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    overflow-x: auto;
  }}
  svg.topology {{
    display: block;
    width: 100%;
    height: auto;
    max-width: {svg_w}px;
  }}

  /* nodes */
  .node {{ cursor: pointer; }}
  .node .cell {{
    stroke: rgba(255,255,255,0.10);
    stroke-width: 1;
    transition: stroke 0.15s, stroke-width 0.15s, filter 0.15s;
  }}
  .node .bs {{
    font-size: 10px;
    fill: rgba(255,255,255,0.88);
    font-family: inherit;
    pointer-events: none;
  }}
  .node .dens {{
    font-size: 9px;
    fill: rgba(255,255,255,0.55);
    font-family: inherit;
    pointer-events: none;
  }}
  .node:hover .cell {{
    stroke: var(--accent);
    stroke-width: 2;
    filter: brightness(1.25);
  }}
  .node.active .cell {{
    stroke: var(--accent-2);
    stroke-width: 2.5;
  }}

  /* sidebar */
  .sidebar {{
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }}
  .card h2 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--muted);
    margin-bottom: 12px;
  }}

  #detail .empty-msg {{
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    padding: 20px 0;
  }}
  #detail .bitstring {{
    font-size: 18px;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 4px;
  }}
  #detail .idx {{
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 14px;
  }}
  #detail .metric-row {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
  }}
  #detail .metric-row:last-of-type {{ border-bottom: none; }}
  #detail .metric-row .k {{ color: var(--muted); }}
  #detail .metric-row .v {{ color: var(--text); font-weight: 600; }}
  #detail .energy-track {{
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    margin: 10px 0 16px;
    overflow: hidden;
  }}
  #detail .energy-track .fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    border-radius: 3px;
    transition: width 0.25s;
  }}
  #detail .files-title {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin: 14px 0 8px;
  }}
  #detail .fp {{
    font-size: 11px;
    color: var(--text);
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 5px 8px;
    margin-bottom: 4px;
    word-break: break-all;
  }}
  #detail .fp.empty {{
    color: var(--muted);
    font-style: italic;
    text-align: center;
  }}

  /* legend */
  .legend-bar {{
    display: flex;
    height: 14px;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 6px;
  }}
  .legend-bar .stop {{ flex: 1; }}
  .legend-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--muted);
  }}

  /* tooltip */
  #tooltip {{
    position: fixed;
    pointer-events: none;
    background: rgba(20,20,31,0.97);
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 11px;
    color: var(--text);
    max-width: 280px;
    z-index: 1000;
    opacity: 0;
    transition: opacity 0.12s;
    box-shadow: 0 6px 24px rgba(0,0,0,0.5);
  }}
  #tooltip.show {{ opacity: 1; }}
  #tooltip .tt-bs {{
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 4px;
  }}
  #tooltip .tt-row {{
    color: var(--muted);
    margin-bottom: 2px;
  }}
  #tooltip .tt-row b {{ color: var(--text); font-weight: 600; }}
  #tooltip .tt-files {{
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
    max-height: 120px;
    overflow-y: auto;
  }}
  #tooltip .tt-files .f {{
    color: var(--accent-2);
    font-size: 10px;
    word-break: break-all;
    margin-bottom: 2px;
  }}

  footer {{
    margin-top: 30px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 10px;
    text-align: center;
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>6-bit TEBM Topology Dashboard <span class="sub">· 64-node visual matrix</span></h1>
    <div class="meta">
      Generated {generated_at}<br />
      Source: topology_map.json
    </div>
  </header>

  <section class="stats">
    <div class="stat">
      <div class="label">Total Nodes</div>
      <div class="value">{TOTAL_NODES}</div>
    </div>
    <div class="stat">
      <div class="label">Mapped Files</div>
      <div class="value warm">{total_files}</div>
    </div>
    <div class="stat">
      <div class="label">Active Edges</div>
      <div class="value">{total_edges}</div>
    </div>
    <div class="stat">
      <div class="label">Avg Energy</div>
      <div class="value">{avg_energy:.4f}</div>
    </div>
  </section>

  <div class="main">
    <div class="canvas">
      <svg class="topology" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg">
        {edges_svg}
        {nodes_svg}
      </svg>
    </div>

    <aside class="sidebar">
      <div class="card" id="detail">
        <h2>Node Inspector</h2>
        <div class="empty-msg">Hover or click a node to inspect its file pathways and local energy.</div>
      </div>

      <div class="card">
        <h2>File Density</h2>
        <div class="legend-bar">{legend_html}</div>
        <div class="legend-labels">
          <span>0 files</span>
          <span>{max_density} files</span>
        </div>
      </div>

      <div class="card">
        <h2>Edge Weight</h2>
        <div class="legend-bar">
          <span class="stop" style="background:rgba(120,170,255,0.10)"></span>
          <span class="stop" style="background:rgba(120,170,255,0.25)"></span>
          <span class="stop" style="background:rgba(120,170,255,0.40)"></span>
          <span class="stop" style="background:rgba(120,170,255,0.55)"></span>
          <span class="stop" style="background:rgba(120,170,255,0.70)"></span>
        </div>
        <div class="legend-labels">
          <span>0.0</span>
          <span>1.0</span>
        </div>
      </div>
    </aside>
  </div>

  <footer>
    6-bit Topological Energy-Based Model · Visual Tracking Interface ·
    Self-contained static HTML — no external dependencies
  </footer>
</div>

<div id="tooltip"></div>

<script>
(function () {{
  var NODES = {nodes_json_str};
  var tooltip = document.getElementById('tooltip');
  var detail  = document.getElementById('detail');
  var activeIndex = null;

  function renderDetail(n) {{
    var filesHtml = n.files.length
      ? n.files.map(function (f) {{
          return '<div class="fp">' + f + '</div>';
        }}).join('')
      : '<div class="fp empty">— no files mapped —</div>';

    detail.innerHTML =
      '<h2>Node Inspector</h2>' +
      '<div class="bitstring">' + n.bitstring + '</div>' +
      '<div class="idx">index ' + n.index + ' · row ' + n.row + ' · col ' + n.col + '</div>' +
      '<div class="metric-row"><span class="k">File density</span><span class="v">' + n.density + '</span></div>' +
      '<div class="metric-row"><span class="k">Local energy</span><span class="v">' + n.energy.toFixed(4) + '</span></div>' +
      '<div class="energy-track"><div class="fill" style="width:' + (n.energy / {max_energy:.6f} * 100) + '%"></div></div>' +
      '<div class="files-title">File Pathways (' + n.density + ')</div>' +
      filesHtml;
  }}

  function showTooltip(n, evt) {{
    var filesPreview = n.files.slice(0, 4).map(function (f) {{
      return '<div class="f">' + f + '</div>';
    }}).join('');
    if (n.files.length > 4) {{
      filesPreview += '<div class="f" style="opacity:0.6">… +' + (n.files.length - 4) + ' more</div>';
    }}
    tooltip.innerHTML =
      '<div class="tt-bs">' + n.bitstring + '</div>' +
      '<div class="tt-row">index: <b>' + n.index + '</b></div>' +
      '<div class="tt-row">density: <b>' + n.density + '</b> files</div>' +
      '<div class="tt-row">energy: <b>' + n.energy.toFixed(4) + '</b></div>' +
      (filesPreview ? '<div class="tt-files">' + filesPreview + '</div>' : '');
    tooltip.classList.add('show');
    moveTooltip(evt);
  }}

  function moveTooltip(evt) {{
    var x = evt.clientX + 14;
    var y = evt.clientY + 14;
    var tw = tooltip.offsetWidth;
    var th = tooltip.offsetHeight;
    if (x + tw > window.innerWidth)  x = evt.clientX - tw - 14;
    if (y + th > window.innerHeight) y = evt.clientY - th - 14;
    tooltip.style.left = x + 'px';
    tooltip.style.top  = y + 'px';
  }}

  function hideTooltip() {{
    tooltip.classList.remove('show');
  }}

  function setActive(i) {{
    document.querySelectorAll('.node.active').forEach(function (el) {{
      el.classList.remove('active');
    }});
    if (i !== null) {{
      var el = document.querySelector('.node[data-i="' + i + '"]');
      if (el) el.classList.add('active');
      activeIndex = i;
      renderDetail(NODES[i]);
    }}
  }}

  document.querySelectorAll('.node').forEach(function (g) {{
    var i = parseInt(g.getAttribute('data-i'), 10);
    var n = NODES[i];

    g.addEventListener('mouseenter', function (evt) {{
      showTooltip(n, evt);
    }});
    g.addEventListener('mousemove', function (evt) {{
      moveTooltip(evt);
    }});
    g.addEventListener('mouseleave', function () {{
      hideTooltip();
    }});
    g.addEventListener('click', function () {{
      setActive(i);
    }});
  }});

  // Auto-select the highest-energy node on load
  var hottest = 0;
  for (var i = 1; i < NODES.length; i++) {{
    if (NODES[i].energy > NODES[hottest].energy) hottest = i;
  }}
  setActive(hottest);
}})();
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    print(f"[dashboard] wrote {output_path}  ({len(html_doc):,} bytes)")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    topology = load_topology_map()
    generate_topology_html(topology)


if __name__ == "__main__":
    main()
