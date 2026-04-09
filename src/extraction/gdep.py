"""
Step 2 — Data Dependency Graph (G_dep).
Spec: Section 3.2 and Section 4.4.2.

Builds G_dep: a bipartite directed graph where edge (n, s) exists if
state variable s is read before or written after node n (call site or function).
"""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def build_gdep(cfg: dict, call_sites: list[dict], state_vars: list[dict]) -> nx.DiGraph:
    """
    Build the data dependency graph G_dep.
    Spec: Section 3.2 — Data Dependency Graph G_dep.

    G_dep is a bipartite directed graph:
      - Left nodes: call site identifiers (call:<function>:<line>)
      - Right nodes: state variable identifiers (var:<name>)
      - Edge (c, s) exists if:
        (a) s is read before call site c within the same function, OR
        (b) s is written after call site c within the same function

    Args:
        cfg: dict mapping function_name -> list of CFG node dicts (from Step 1)
        call_sites: list of call site dicts (from Step 1)
        state_vars: list of state variable dicts (from Step 1)

    Returns:
        G_dep: nx.DiGraph with call site and state variable nodes
    """
    G_dep = nx.DiGraph()

    # Add state variable nodes
    var_names = {v["name"] for v in state_vars}
    for var_name in var_names:
        G_dep.add_node(f"var:{var_name}", node_type="state_var")

    # For each call site, find dependent state variables
    for cs in call_sites:
        func_name = cs["function"]
        call_line = cs["line"]
        call_id = f"call:{func_name}:{call_line}"

        G_dep.add_node(call_id, node_type="call_site")

        if func_name not in cfg:
            logger.warning("Function %s not found in CFG", func_name)
            continue

        cfg_nodes = cfg[func_name]

        # Find the index of the external call node in the CFG
        call_idx = _find_call_node_index(cfg_nodes, call_line)
        if call_idx is None:
            logger.warning(
                "External call node not found in CFG for %s at line %d",
                func_name,
                call_line,
            )
            continue

        # S(c) = vars read before c OR written after c
        dependent_vars = set()

        # (a) State vars read before the call
        for i in range(call_idx):
            for var_name in cfg_nodes[i]["state_vars_read"]:
                if var_name in var_names:
                    dependent_vars.add(var_name)

        # (b) State vars written after the call
        for i in range(call_idx + 1, len(cfg_nodes)):
            for var_name in cfg_nodes[i]["state_vars_written"]:
                if var_name in var_names:
                    dependent_vars.add(var_name)

        # Add edges
        for var_name in dependent_vars:
            G_dep.add_edge(call_id, f"var:{var_name}")

    return G_dep


def _find_call_node_index(cfg_nodes: list[dict], call_line: int) -> int | None:
    """
    Find the index of the external call CFG node matching the given line.
    Returns the first CFG node with has_external_call=True on that line.
    """
    for i, node in enumerate(cfg_nodes):
        if node["has_external_call"] and node["line"] == call_line:
            return i
    # Fallback: match by line only (some calls span multiple nodes)
    for i, node in enumerate(cfg_nodes):
        if node["has_external_call"]:
            return i
    return None
