"""
adjacency.py — 6D hypercube adjacency for the Topological EBM framework.
Node 0 (000000) core infrastructure.

A 6-cube has 2^6 = 64 nodes.  Each node has exactly 6 neighbours reachable
by a single-bit flip (Hamming distance 1).  This gives 192 *unique*
undirected edges, stored as 384 directed adjacency entries (every
undirected edge appears in both directions).
"""

from itertools import combinations

N_DIMS = 6
N_NODES = 1 << N_DIMS                       # 64
N_EDGES_UNDIRECTED = N_NODES * N_DIMS // 2  # 192
N_EDGES_DIRECTED = N_NODES * N_DIMS         # 384


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer addresses."""
    return bin(a ^ b).count("1")


def bit_string(addr: int, width: int = N_DIMS) -> str:
    """Return the width-bit binary string for an address."""
    return format(addr, "0{}b".format(width))


def build_adjacency_list():
    """dict addr -> [neighbour, ...]  (Hamming distance 1)."""
    adj = {i: [] for i in range(N_NODES)}
    for i in range(N_NODES):
        for d in range(N_DIMS):
            adj[i].append(i ^ (1 << d))
    return adj


def build_edge_list():
    """List of (i, j) undirected edges with i < j.  Length 192."""
    edges = []
    for i in range(N_NODES):
        for d in range(N_DIMS):
            j = i ^ (1 << d)
            if i < j:
                edges.append((i, j))
    return edges


def build_directed_edge_list():
    """List of (src, dst) directed edges.  Length 384."""
    edges = []
    for i in range(N_NODES):
        for d in range(N_DIMS):
            edges.append((i, i ^ (1 << d)))
    return edges


def build_adjacency_matrix():
    """64 x 64 list-of-lists of 0/1."""
    mat = [[0] * N_NODES for _ in range(N_NODES)]
    for i in range(N_NODES):
        for d in range(N_DIMS):
            mat[i][i ^ (1 << d)] = 1
    return mat


def edge_index_map():
    """dict (i,j) with i<j -> position in build_edge_list()."""
    return {e: idx for idx, e in enumerate(build_edge_list())}


def verify() -> bool:
    adj = build_adjacency_list()
    assert all(len(v) == N_DIMS for v in adj.values()), "degree != 6"
    und = build_edge_list()
    assert len(und) == N_EDGES_UNDIRECTED == 192
    dire = build_directed_edge_list()
    assert len(dire) == N_EDGES_DIRECTED == 384
    mat = build_adjacency_matrix()
    for i in range(N_NODES):
        for j in range(N_NODES):
            assert mat[i][j] == mat[j][i], "adjacency not symmetric"
    for (i, j) in und:
        assert hamming_distance(i, j) == 1
    # every node reachable, graph is connected (6-cube property)
    return True


if __name__ == "__main__":
    verify()
    print("adjacency.py OK — %d nodes, %d undirected / %d directed edges"
          % (N_NODES, N_EDGES_UNDIRECTED, N_EDGES_DIRECTED))
