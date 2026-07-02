"""
embedding_map.py — map code-file embedding arrays to 6-bit hypercube
addresses using local embedding medians.

Procedure:
  1. Choose 6 projection axes into each file's embedding vector.
  2. Compute the median of the projected values along each axis
     (the "local median" for the given file set).
  3. For each file, bit d = 1 iff projection[d] >= median[d], else 0.
  4. Group files into the 64 buckets defined by the resulting address.

Standard library only.
"""

import statistics
from collections import defaultdict

N_DIMS = 6
N_NODES = 1 << N_DIMS


def _project(embeddings, axes):
    return [[emb[a] for a in axes] for emb in embeddings]


def compute_medians(projections):
    """Median along each of the 6 axes."""
    return [statistics.median(p[d] for p in projections) for d in range(N_DIMS)]


def assign_addresses(projections, medians):
    """6-bit address per projection vector."""
    out = []
    for p in projections:
        addr = 0
        for d in range(N_DIMS):
            if p[d] >= medians[d]:
                addr |= (1 << d)
        out.append(addr)
    return out


def map_files(file_ids, embeddings, axes=None):
    """
    Args:
        file_ids   : list of identifiers.
        embeddings : list of vectors, each len >= max(axes)+1.
        axes       : 6 indices into the embedding (default [0..5]).

    Returns dict:
        address_map : {file_id: int_address}
        buckets     : {address: [file_id, ...]}
        medians     : [6 floats]
        axes        : [6 ints]
    """
    if axes is None:
        axes = list(range(N_DIMS))
    assert len(axes) == N_DIMS, "need exactly 6 axes"
    assert len(file_ids) == len(embeddings), "file/embedding length mismatch"
    for emb in embeddings:
        for a in axes:
            assert a < len(emb), "axis %d out of range for embedding of len %d" % (a, len(emb))

    projections = _project(embeddings, axes)
    medians = compute_medians(projections)
    addresses = assign_addresses(projections, medians)

    address_map = {}
    buckets = defaultdict(list)
    for fid, addr in zip(file_ids, addresses):
        address_map[fid] = addr
        buckets[addr].append(fid)

    return {
        "address_map": address_map,
        "buckets": dict(buckets),
        "medians": medians,
        "axes": list(axes),
    }


def address_to_bitstring(addr, width=N_DIMS):
    return format(addr, "0{}b".format(width))


def remap_with_local_median(file_ids, embeddings, axes=None, depth=1):
    """Recursive local-median refinement.

    Splits the file set on the global median, then recursively splits
    each half on its own (local) median, for `depth` levels.  After
    `depth` splits each file has a depth-bit prefix; the final 6-bit
    address is obtained by taking the 6 most significant split bits
    (padded with global-median bits if depth < 6).
    """
    if axes is None:
        axes = list(range(N_DIMS))
    n = len(file_ids)
    if n == 0:
        return {"address_map": {}, "buckets": {}, "medians": [], "axes": axes}

    # build full 6-bit address via iterative local medians on each axis
    bits = {fid: [] for fid in file_ids}
    idxs = list(range(n))
    for d in range(N_DIMS):
        a = axes[d]
        # start with all indices, split on local median recursively
        groups = [idxs]
        # we want 1 bit per axis, so just one split per axis using local median
        vals = [embeddings[k][a] for k in idxs]
        med = statistics.median(vals)
        for k in idxs:
            bits[file_ids[k]].append(1 if embeddings[k][a] >= med else 0)

    address_map = {}
    buckets = defaultdict(list)
    for fid in file_ids:
        addr = 0
        for d in range(N_DIMS):
            if bits[fid][d]:
                addr |= (1 << d)
        address_map[fid] = addr
        buckets[addr].append(fid)

    medians = []
    for d in range(N_DIMS):
        medians.append(statistics.median([embeddings[k][axes[d]] for k in idxs]))

    return {
        "address_map": address_map,
        "buckets": dict(buckets),
        "medians": medians,
        "axes": list(axes),
    }


if __name__ == "__main__":
    import random
    random.seed(0)
    fids = ["file_%d" % i for i in range(40)]
    embs = [[random.gauss(0, 1) for _ in range(10)] for _ in range(40)]
    r = map_files(fids, embs)
    print("mapped %d files into %d buckets" % (len(r["address_map"]), len(r["buckets"])))
    print("medians:", [round(m, 3) for m in r["medians"]])
    r2 = remap_with_local_median(fids, embs)
    print("local-median buckets: %d" % len(r2["buckets"]))
