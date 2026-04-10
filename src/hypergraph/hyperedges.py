"""
Step 5 — Hyperedge Construction and Incidence Matrix (H).
Spec: Section 4.5–4.6 (Hyperedge Definition + Algorithm).

For each call site c in V_c, constructs hyperedge e_c = {c} ∪ F(c) ∪ S(c).
Then builds binary incidence matrix H_inc.
"""

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def bounded_ancestors(G: nx.DiGraph, node: str, delta: int = 3) -> set[str]:
    """
    Return ancestors of node reachable within delta hops via BFS.
    Spec: Section 4.5 — F(c) uses bounded ancestor expansion.

    Args:
        G: directed graph (G_call)
        node: starting node
        delta: maximum depth bound (default 3)

    Returns:
        Set of ancestor node names (excluding the starting node itself)
    """
    if node not in G:
        return set()

    visited = set()
    frontier = {node}
    for _ in range(delta):
        next_frontier = set()
        for n in frontier:
            for pred in G.predecessors(n):
                if pred not in visited and pred != node:
                    visited.add(pred)
                    next_frontier.add(pred)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def build_hyperedges(
    V: list[str],
    V_c: list[str],
    V_s: list[str],
    node_index: dict[str, int],
    G_call: nx.DiGraph,
    G_dep: nx.DiGraph,
    call_sites: list[dict],
    delta: int = 3,
) -> tuple[list[set[str]], np.ndarray]:
    """
    Construct hyperedges and incidence matrix.
    Spec: Section 4.5–4.6 — Hyperedge Construction Algorithm.

    For each call site c in V_c:
        e_c = {c} ∪ F(c) ∪ S(c)
    where:
        F(c) = {f} ∪ BoundedAncestors(f, G_call, delta)
        S(c) = {s in V_s | (c, s) in G_dep}

    Args:
        V: combined node list
        V_c: call site node identifiers
        V_s: state variable node identifiers
        node_index: maps node identifier to integer index in V
        G_call: call graph (nx.DiGraph)
        G_dep: data dependency graph (nx.DiGraph)
        call_sites: call site dicts from Step 1 (to map call node → function name)
        delta: ancestor depth bound (default 3)

    Returns:
        E: list of hyperedges (each a set of node identifiers)
        H_inc: binary incidence matrix of shape (|V|, |E|), dtype float32
    """
    set_s = set(V_s)
    n_nodes = len(V)
    n_edges = len(V_c)

    # Build lookup: call site node id → function name
    cs_to_func = {}
    for cs in call_sites:
        cs_id = f"call:{cs['function']}:{cs['line']}"
        cs_to_func[cs_id] = cs["function"]

    E = []
    H_inc = np.zeros((n_nodes, n_edges), dtype=np.float32)

    for j, c in enumerate(V_c):
        # Start with the call site itself
        e_c = {c}

        # F(c) = {f} ∪ BoundedAncestors(f, G_call, delta)
        func_name = cs_to_func.get(c)
        if func_name:
            func_node = f"func:{func_name}"
            if func_node in node_index:
                e_c.add(func_node)
            # Add bounded ancestors
            ancestors = bounded_ancestors(G_call, func_name, delta)
            for anc in ancestors:
                anc_node = f"func:{anc}"
                if anc_node in node_index:
                    e_c.add(anc_node)

        # S(c) = {s in V_s | (c, s) in G_dep}
        if c in G_dep:
            for successor in G_dep.successors(c):
                if successor in set_s and successor in node_index:
                    e_c.add(successor)

        E.append(e_c)

        # Fill incidence matrix column
        for node in e_c:
            if node in node_index:
                H_inc[node_index[node], j] = 1.0

    # Assertions
    assert len(E) == len(V_c), f"|E|={len(E)} != |V_c|={len(V_c)}"
    assert H_inc.shape == (n_nodes, n_edges)
    assert np.all((H_inc == 0) | (H_inc == 1)), "H_inc must be binary"

    # Every call site must appear in its own hyperedge
    for j, c in enumerate(V_c):
        if c in node_index:
            assert H_inc[node_index[c], j] == 1.0, f"Call site {c} not in its own hyperedge"

    return E, H_inc
