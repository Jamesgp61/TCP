"""
energy_model.py — energy calculation and Gibbs-sampling relaxation engine
for the Topological Energy-Based Model on the 6-bit hypercube.

State :  s_i in {0, 1}   for i = 0..63
Spin  :  x_i = 2*s_i - 1 in {-1, +1}

Energy:
    E(s) = - sum_{(i,j) in edges} J_ij * x_i * x_j
           - sum_i h_i * x_i

Gibbs update of node i:
    eff_i = h_i + sum_{j in N(i)} J_ij * x_j
    P(s_i = 1) = sigmoid(2 * eff_i / T)

Standard library only.
"""

import math
import random

from adjacency import (
    N_DIMS,
    N_NODES,
    build_adjacency_list,
    build_edge_list,
)

_ADJ = build_adjacency_list()
_EDGES = build_edge_list()


# ── helpers ───────────────────────────────────────────────────────────

def spins_from_states(states):
    return [2 * s - 1 for s in states]


def states_from_spins(spins):
    return [(x + 1) // 2 for x in spins]


def _edge_key(i, j):
    return (i, j) if i < j else (j, i)


# ── energy ────────────────────────────────────────────────────────────

def compute_energy(states, J=None, h=None):
    """Total energy of a 64-node configuration."""
    spins = spins_from_states(states)
    if h is None:
        h = [0.0] * N_NODES
    energy = 0.0
    if J is not None:
        for (i, j) in _EDGES:
            energy -= J.get(_edge_key(i, j), 0.0) * spins[i] * spins[j]
    for i in range(N_NODES):
        energy -= h[i] * spins[i]
    return energy


def local_field(states, i, J=None, h=None):
    """Effective field on node i given current neighbour states."""
    if h is None:
        h = [0.0] * N_NODES
    eff = h[i]
    if J is None:
        return eff
    spins = spins_from_states(states)
    for j in _ADJ[i]:
        eff += J.get(_edge_key(i, j), 0.0) * spins[j]
    return eff


# ── Gibbs sampling ────────────────────────────────────────────────────

def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def gibbs_update(states, i, J=None, h=None, temperature=1.0):
    """In-place single-site Gibbs update of node i."""
    eff = local_field(states, i, J, h)
    p1 = _sigmoid(2.0 * eff / max(temperature, 1e-12))
    states[i] = 1 if random.random() < p1 else 0
    return states


def gibbs_sweep(states, J=None, h=None, temperature=1.0, order=None):
    """One full sweep (random order by default)."""
    if order is None:
        order = list(range(N_NODES))
        random.shuffle(order)
    for i in order:
        gibbs_update(states, i, J, h, temperature)
    return states


def relax(states, J=None, h=None, n_sweeps=100,
          t_start=2.0, t_end=0.1, seed=None, track_energy=False):
    """Simulated-annealing-style Gibbs relaxation.

    Temperature decays geometrically from t_start to t_end over n_sweeps.
    Returns dict with final states, energy, and optional history.
    """
    if seed is not None:
        random.seed(seed)
    states = list(states)
    history = []
    n = max(n_sweeps, 1)
    for sweep in range(n):
        frac = sweep / max(n - 1, 1)
        T = t_start * (t_end / t_start) ** frac
        gibbs_sweep(states, J, h, temperature=T)
        if track_energy:
            history.append(compute_energy(states, J, h))
    return {
        "states": states,
        "energy": compute_energy(states, J, h),
        "history": history if track_energy else None,
    }


# ── factory helpers ───────────────────────────────────────────────────

def random_initial_state(seed=None):
    if seed is not None:
        random.seed(seed)
    return [random.randint(0, 1) for _ in range(N_NODES)]


def default_couplings(strength=1.0):
    """Uniform coupling on every edge."""
    return {_edge_key(i, j): strength for (i, j) in _EDGES}


def default_fields(value=0.0):
    return [value] * N_NODES


def random_couplings(seed=None, scale=1.0):
    if seed is not None:
        random.seed(seed)
    return {_edge_key(i, j): random.gauss(0, scale) for (i, j) in _EDGES}


# ── self-test ─────────────────────────────────────────────────────────

def verify():
    s0 = random_initial_state(seed=1)
    J = default_couplings(0.5)
    h = default_fields(0.1)
    e0 = compute_energy(s0, J, h)
    res = relax(s0, J, h, n_sweeps=50, t_start=2.0, t_end=0.05,
                seed=1, track_energy=True)
    assert len(res["states"]) == N_NODES
    assert len(res["history"]) == 50
    # energy should be a float
    assert isinstance(res["energy"], float)
    # local_field sanity: with no couplings equals h[i]
    lf = local_field([0] * N_NODES, 0, J=None, h=[0.3] * N_NODES)
    assert abs(lf - 0.3) < 1e-9
    return True


if __name__ == "__main__":
    verify()
    J = default_couplings(0.5)
    h = default_fields(0.1)
    s0 = random_initial_state(seed=42)
    res = relax(s0, J, h, n_sweeps=200, t_start=2.0, t_end=0.05,
                seed=42, track_energy=True)
    print("initial energy: %.4f" % res["history"][0])
    print("final energy:   %.4f" % res["energy"])
    print("delta:          %.4f" % (res["history"][0] - res["energy"]))
    print("nodes ON:       %d / %d" % (sum(res["states"]), N_NODES))
