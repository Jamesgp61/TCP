#!/usr/bin/env python3
"""
edge_trainer.py
===============

Core learning-feedback layer for the open-source 6-bit Topological
Energy-Based Model (TEM-6) framework.

Responsibilities
----------------
1.  Parse a development-run log (file path or in-memory list) and extract
    the ordered sequence of 6-bit Node IDs that were exercised.
2.  Accept a binary status code — ``True`` for success/compilation,
    ``False`` for crash/test-failure — and feed it into the update rule.
3.  Apply a **symmetric, Hebbian-style weight adjustment** to the 384
    bidirectional edges stored in ``topology_map.json``.

    * 384 directed entries = 192 unique undirected pairs.
    * Each of the 64 possible 6-bit nodes has exactly 6 neighbours at
      Hamming Distance 1  →  64 × 6 = 384 directed edges.
    * The learning rule is an energy-relaxation step function:

          Δw = η·s·(1 − s·w) − λ·w

      where  s = +1 (success) or −1 (failure),
             η = learning rate,
             λ = decay (relaxation toward zero energy).

    * Edge updates are **strictly** constrained to node pairs whose
      Hamming Distance ≤ 1.  All other edges are left untouched.
    * Node keys present in ``topology_map.json`` are preserved verbatim.

Portability
-----------
Only ``json``, ``os``, and ``sys`` from the standard library are used.
No third-party dependencies.  Tested on CPython 3.6+.
"""

import json
import os
import sys

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

NODE_ID_BITS = 6
TOTAL_NODES = 1 << NODE_ID_BITS          # 64
EDGES_PER_NODE = NODE_ID_BITS            # 6 neighbours at HD=1
TOTAL_DIRECTED_EDGES = TOTAL_NODES * EDGES_PER_NODE   # 384

MAX_HAMMING_DISTANCE = 1

DEFAULT_LEARNING_RATE = 0.05
DEFAULT_DECAY_RATE = 0.01
WEIGHT_FLOOR = -1.0
WEIGHT_CEIL = 1.0

DEFAULT_TOPOLOGY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "topology_map.json",
)

# ──────────────────────────────────────────────────────────────────────
# Low-level utilities
# ──────────────────────────────────────────────────────────────────────


def _validate_node_id(node_id):
    """Return *True* iff *node_id* is a valid 6-bit binary string."""
    if not isinstance(node_id, str):
        return False
    if len(node_id) != NODE_ID_BITS:
        return False
    return all(ch in "01" for ch in node_id)


def _hamming_distance(a, b):
    """Hamming distance between two equal-length binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _clamp(value, low, high):
    """Clamp *value* to the closed interval [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _all_node_ids():
    """Generate every possible 6-bit node ID, zero-padded."""
    return [format(i, "0{}b".format(NODE_ID_BITS)) for i in range(TOTAL_NODES)]


def _neighbours(node_id):
    """Return the 6 node IDs at Hamming Distance exactly 1 from *node_id*."""
    result = []
    for i in range(NODE_ID_BITS):
        flipped = list(node_id)
        flipped[i] = "1" if node_id[i] == "0" else "0"
        result.append("".join(flipped))
    return result


# ──────────────────────────────────────────────────────────────────────
# Topology initialisation, loading, saving
# ──────────────────────────────────────────────────────────────────────


def init_topology():
    """Build a fresh topology dict with all 384 directed edges at weight 0.

    Schema::

        {
            "nodes": { "000000": {}, "000001": {}, … },
            "edges": { "000000": {"000001": 0.0, …}, … }
        }
    """
    all_ids = _all_node_ids()
    nodes = {nid: {} for nid in all_ids}
    edges = {}
    for nid in all_ids:
        edges[nid] = {}
        for nbr in _neighbours(nid):
            edges[nid][nbr] = 0.0
    return {"nodes": nodes, "edges": edges}


def load_topology(path):
    """Load and validate ``topology_map.json`` from *path*."""
    if not os.path.isfile(path):
        raise FileNotFoundError("Topology file not found: {}".format(path))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            topology = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in {}: {}".format(path, exc))

    if not isinstance(topology, dict):
        raise ValueError("Topology root must be a JSON object.")
    if "edges" not in topology or not isinstance(topology["edges"], dict):
        raise ValueError("Topology must contain an 'edges' object.")
    return topology


def save_topology(topology, path):
    """Atomically write *topology* to *path* (temp-file + os.replace)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(topology, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ──────────────────────────────────────────────────────────────────────
# Log / execution-list parsing  (Requirement 1)
# ──────────────────────────────────────────────────────────────────────


def parse_log(log_source):
    """Parse a log file or in-memory list into an ordered list of node IDs.

    Parameters
    ----------
    log_source : str or list
        * If a *str* that is an existing file path, the file is read line by
          line.  Tokens may be whitespace- or comma-separated.
        * If a *list* / *tuple*, each element is treated as a token (strings
          are split on whitespace/commas).

    Returns
    -------
    list[str]
        Validated 6-bit node IDs in order of first appearance.
    """
    if isinstance(log_source, (list, tuple)):
        raw = []
        for item in log_source:
            if isinstance(item, str):
                raw.extend(item.replace(",", " ").split())
            else:
                raw.append(str(item))
    elif isinstance(log_source, str) and os.path.isfile(log_source):
        raw = []
        with open(log_source, "r", encoding="utf-8") as fh:
            for line in fh:
                raw.extend(line.replace(",", " ").split())
    else:
        raise ValueError(
            "log_source must be a file path or a list of node IDs; "
            "received {!r}".format(type(log_source).__name__)
        )

    node_ids = []
    for token in raw:
        if _validate_node_id(token):
            node_ids.append(token)
        else:
            sys.stderr.write(
                "WARNING: ignoring invalid node ID {!r}\n".format(token)
            )
    return node_ids


# ──────────────────────────────────────────────────────────────────────
# Symmetric weight-adjustment algorithm  (Requirements 2 & 3)
# ──────────────────────────────────────────────────────────────────────


def _ensure_edge(edges, src, dst):
    """Return current weight for *src*→*dst*, creating the slot if needed."""
    bucket = edges.get(src)
    if bucket is None:
        bucket = {}
        edges[src] = bucket
    if dst not in bucket:
        bucket[dst] = 0.0
    return bucket[dst]


def update_edges(topology, node_ids, status,
                 lr=DEFAULT_LEARNING_RATE,
                 decay=DEFAULT_DECAY_RATE):
    """Apply one energy-relaxation step to the topology's edge weights.

    For every **unique pair** of nodes that co-occurred in *node_ids* and
    whose Hamming Distance ≤ 1, the bidirectional edge weight is updated:

        s = +1  if status is True   (success → potentiation)
        s = −1  if status is False  (failure → depression)

        Δw = η · s · (1 − s·w) − λ · w

    The first term is the Hebbian co-activation signal; the second term
    (−λ·w) is the energy-relaxation decay that drives every touched edge
    toward zero, preventing unbounded growth.

    Both directions (a→b and b→a) receive the **same** Δw, keeping the
    edge matrix symmetric.

    Node keys in ``topology["nodes"]`` are never modified.
    """
    edges = topology["edges"]

    # Unique nodes in order of first appearance.
    active = list(dict.fromkeys(node_ids))

    # Enumerate eligible pairs (Hamming Distance ≤ 1, i < j).
    eligible = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            if _hamming_distance(a, b) <= MAX_HAMMING_DISTANCE:
                eligible.append((a, b))

    if not eligible:
        return topology

    s = 1.0 if status else -1.0

    for a, b in eligible:
        w_ab = _ensure_edge(edges, a, b)
        w_ba = _ensure_edge(edges, b, a)

        delta = lr * s * (1.0 - s * w_ab) - decay * w_ab
        new_ab = _clamp(w_ab + delta, WEIGHT_FLOOR, WEIGHT_CEIL)

        delta = lr * s * (1.0 - s * w_ba) - decay * w_ba
        new_ba = _clamp(w_ba + delta, WEIGHT_FLOOR, WEIGHT_CEIL)

        edges[a][b] = new_ab
        edges[b][a] = new_ba

    return topology


# ──────────────────────────────────────────────────────────────────────
# High-level convenience class
# ──────────────────────────────────────────────────────────────────────


class EdgeTrainer:
    """Stateful wrapper around the three core operations.

    Example
    -------
    >>> trainer = EdgeTrainer("topology_map.json")
    >>> trainer.train("run_42.log", status=True)
    >>> trainer.save()
    """

    def __init__(self, topology_path=None,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 decay_rate=DEFAULT_DECAY_RATE):
        self.topology_path = topology_path or DEFAULT_TOPOLOGY_PATH
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.topology = None

    # -- lifecycle --------------------------------------------------

    def load(self):
        """Load topology from disk, initialising if the file is absent."""
        if os.path.isfile(self.topology_path):
            self.topology = load_topology(self.topology_path)
        else:
            sys.stderr.write(
                "INFO: {} not found — initialising fresh topology.\n".format(
                    self.topology_path
                )
            )
            self.topology = init_topology()
            save_topology(self.topology, self.topology_path)
        return self

    def save(self):
        """Persist the in-memory topology back to disk."""
        if self.topology is None:
            raise RuntimeError("No topology loaded; call load() first.")
        save_topology(self.topology, self.topology_path)
        return self

    # -- main operation --------------------------------------------

    def train(self, log_source, status):
        """Parse *log_source*, apply *status*, and update edge weights.

        Parameters
        ----------
        log_source : str or list
            File path or list of 6-bit node ID strings.
        status : bool
            ``True``  → success / compilation passed.
            ``False`` → crash / test failure.
        """
        if self.topology is None:
            self.load()

        node_ids = parse_log(log_source)
        if not node_ids:
            sys.stderr.write("WARNING: no valid node IDs parsed from log.\n")
            return self

        update_edges(
            self.topology,
            node_ids,
            bool(status),
            lr=self.learning_rate,
            decay=self.decay_rate,
        )
        return self


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────


def _cli():
    """``python edge_trainer.py <topology_map.json> <log> <status>``

    *status* is interpreted as **True** for any of:
    ``success``, ``pass``, ``1``, ``true``, ``ok``, ``yes`` (case-insensitive).
    Any other value is treated as **False**.
    """
    if len(sys.argv) < 4:
        sys.stderr.write(
            "Usage: python edge_trainer.py "
            "<topology_map.json> <log_file_or_list> <success|failure>\n"
        )
        return 1

    topology_path = sys.argv[1]
    log_arg = sys.argv[2]
    status_str = sys.argv[3].strip().lower()
    status = status_str in ("success", "pass", "1", "true", "ok", "yes")

    # Allow comma-separated inline list instead of a file path.
    if os.path.isfile(log_arg):
        log_source = log_arg
    else:
        log_source = log_arg.replace(",", " ").split()

    try:
        trainer = EdgeTrainer(topology_path)
        trainer.train(log_source, status)
        trainer.save()
        sys.stdout.write(
            "Edge training complete (status={}). "
            "Topology written to {}.\n".format(status, topology_path)
        )
    except Exception as exc:  # top-level guard
        sys.stderr.write("ERROR: {}\n".format(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
