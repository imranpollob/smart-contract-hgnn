"""
Step 3 — Node Set Construction (V).
Spec: Section 4.2 (Node Set).

Builds the three disjoint node sets V_f, V_s, V_c and combines them into V.
Each node has a unique string identifier:
  - Functions:       "func:<function_name>"
  - State variables: "var:<variable_name>"
  - Call sites:      "call:<function_name>:<line_number>"
"""

import logging

logger = logging.getLogger(__name__)


def build_node_sets(
    functions: list[dict],
    state_vars: list[dict],
    call_sites: list[dict],
) -> dict:
    """
    Construct disjoint node sets V_f, V_s, V_c and combined V.
    Spec: Section 4.2 — Node Set V = V_f ∪ V_s ∪ V_c.

    Args:
        functions: list of function dicts from Step 1 (extract_function_info)
        state_vars: list of state variable dicts from Step 1 (extract_state_variable_info)
        call_sites: list of call site dicts from Step 1 (extract_external_call_sites)

    Returns:
        dict with keys: V_f, V_s, V_c, V, node_index
    """
    V_f = [f"func:{f['name']}" for f in functions]
    V_s = [f"var:{v['name']}" for v in state_vars]
    V_c = [f"call:{cs['function']}:{cs['line']}" for cs in call_sites]

    # Verify disjointness (spec requirement)
    set_f, set_s, set_c = set(V_f), set(V_s), set(V_c)
    assert set_f.isdisjoint(set_s), f"V_f ∩ V_s is not empty: {set_f & set_s}"
    assert set_f.isdisjoint(set_c), f"V_f ∩ V_c is not empty: {set_f & set_c}"
    assert set_s.isdisjoint(set_c), f"V_s ∩ V_c is not empty: {set_s & set_c}"

    V = V_f + V_s + V_c

    assert len(V) == len(V_f) + len(V_s) + len(V_c)

    node_index = {node: i for i, node in enumerate(V)}

    return {
        "V_f": V_f,
        "V_s": V_s,
        "V_c": V_c,
        "V": V,
        "node_index": node_index,
    }
