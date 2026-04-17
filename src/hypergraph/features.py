"""
Step 4 — Node Feature Matrix (X).
Spec: Section 4.3 (Node Features).

Computes feature vector x_v for each node v in V and assembles into matrix X.
Features are type-specific and zero-padded to a common dimension d.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature schema constants ──────────────────────────────────────

# V_f features: visibility (4) + mutability (4) + is_constructor (1) = 9
VISIBILITY_CLASSES = ["public", "private", "internal", "external"]
MUTABILITY_CLASSES = ["pure", "view", "payable", "nonpayable"]
N_FUNC_FEATURES = len(VISIBILITY_CLASSES) + len(MUTABILITY_CLASSES) + 1  # 9

# V_s features: type_category (8) + normalized_slot (1) + access_pattern (3)
#   + written_after_call (1) + read_before_call (1) = 14
# The last two flags are the reentrancy-specific extensions introduced to
# surface the "write-after-external-call" / "read-before-external-call"
# pattern directly to the model.
TYPE_CATEGORIES = [
    "uint",       # uint256, uint8, int256, etc.
    "address",    # address
    "bool",       # bool
    "bytes",      # bytes32, bytes, string
    "mapping",    # mapping(...)
    "array",      # type[]
    "struct",     # custom struct types
    "other",      # anything else
]
ACCESS_PATTERNS = ["read_only", "write_only", "read_write"]
N_STATE_FEATURES = (
    len(TYPE_CATEGORIES)  # 8
    + 1                   # normalized slot
    + len(ACCESS_PATTERNS)  # 3
    + 1                   # written_after_call (any call site in same function writes this after)
    + 1                   # read_before_call
)  # 14

# V_c features: call_type_opcode (4) + value_transfer (1)
#   + gas_forwarded (1) + sender_controlled_target (1) + guarded_by_modifier (1)
#   + writes_after_call (1 count, log-normalized)
#   + reads_after_call (1 count, log-normalized)
#   + reads_before_call (1 count, log-normalized) = 11
# These per-call-site fields carry the reentrancy pattern signal directly:
# the defining reentrancy condition is a state write that occurs after an
# attacker-influenceable external call that forwards gas, inside a function
# that is not protected by a nonReentrant guard.
CALL_OPCODES = ["call", "delegatecall", "staticcall", "other"]
N_CALL_FEATURES = (
    len(CALL_OPCODES)  # 4
    + 1                # value_transfer
    + 1                # gas_forwarded
    + 1                # sender_controlled_target
    + 1                # guarded_by_modifier
    + 1                # writes_after_call_count (log-normalized)
    + 1                # reads_after_call_count (log-normalized)
    + 1                # reads_before_call_count (log-normalized)
)  # 11

# Total feature dimension d (all padded to this)
FEATURE_DIM = N_FUNC_FEATURES + N_STATE_FEATURES + N_CALL_FEATURES  # 34


def build_feature_matrix(
    V: list[str],
    V_f: list[str],
    V_s: list[str],
    V_c: list[str],
    functions: list[dict],
    state_vars: list[dict],
    call_sites: list[dict],
    cfg: dict,
) -> np.ndarray:
    """
    Compute node feature matrix X.
    Spec: Section 4.3 — Node Features.

    Each node v gets a feature vector x_v in R^d. Features are type-specific:
    - V_f nodes: visibility (one-hot), mutability (one-hot), is_constructor
    - V_s nodes: type category (one-hot), normalized slot, access pattern (one-hot)
    - V_c nodes: call opcode (one-hot), value-transfer flag

    All vectors are zero-padded to the same dimension d.

    Args:
        V: combined node list
        V_f, V_s, V_c: disjoint node sets
        functions: function metadata dicts from Step 1
        state_vars: state variable metadata dicts from Step 1
        call_sites: call site dicts from Step 1
        cfg: CFG dict from Step 1 (used to determine access patterns)

    Returns:
        X: np.ndarray of shape (|V|, d)
    """
    n = len(V)
    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    # Build lookup maps
    func_map = {f"func:{f['name']}": f for f in functions}
    var_map = {f"var:{v['name']}": v for v in state_vars}
    cs_map = {f"call:{cs['function']}:{cs['line']}": cs for cs in call_sites}

    # Compute access patterns for state vars from CFG
    access_patterns = _compute_access_patterns(cfg, state_vars)

    # Compute per-var flags for the two reentrancy-specific V_s features by
    # aggregating across all call sites in the contract: is this state var
    # ever written *after* an external call, or read *before* one?
    written_after_call, read_before_call = _compute_state_var_reentrancy_flags(
        call_sites, state_vars
    )

    # Compute max slot for normalization
    max_slot = max((v["slot"] for v in state_vars), default=1)
    if max_slot == 0:
        max_slot = 1

    set_f = set(V_f)
    set_s = set(V_s)
    set_c = set(V_c)

    for i, node in enumerate(V):
        if node in set_f:
            X[i] = _encode_function(func_map.get(node), FEATURE_DIM)
        elif node in set_s:
            var_name = node.replace("var:", "", 1)
            X[i] = _encode_state_var(
                var_map.get(node),
                access_patterns.get(var_name, "read_write"),
                max_slot,
                FEATURE_DIM,
                written_after_call=written_after_call.get(var_name, False),
                read_before_call=read_before_call.get(var_name, False),
            )
        elif node in set_c:
            X[i] = _encode_call_site(cs_map.get(node), FEATURE_DIM)

    # Sanity checks
    assert X.shape == (n, FEATURE_DIM)
    assert not np.any(np.isnan(X)), "NaN in feature matrix"
    assert not np.any(np.isinf(X)), "Inf in feature matrix"

    return X


def get_feature_config() -> dict:
    """Return the feature schema configuration for reproducibility."""
    return {
        "feature_dim": FEATURE_DIM,
        "func_features": {
            "offset": 0,
            "size": N_FUNC_FEATURES,
            "visibility_classes": VISIBILITY_CLASSES,
            "mutability_classes": MUTABILITY_CLASSES,
        },
        "state_var_features": {
            "offset": N_FUNC_FEATURES,
            "size": N_STATE_FEATURES,
            "type_categories": TYPE_CATEGORIES,
            "access_patterns": ACCESS_PATTERNS,
            "reentrancy_flags": [
                "written_after_call",
                "read_before_call",
            ],
        },
        "call_site_features": {
            "offset": N_FUNC_FEATURES + N_STATE_FEATURES,
            "size": N_CALL_FEATURES,
            "call_opcodes": CALL_OPCODES,
            "reentrancy_flags": [
                "value_transfer",
                "gas_forwarded",
                "sender_controlled_target",
                "guarded_by_modifier",
            ],
            "reentrancy_counts": [
                "writes_after_call_count_log",
                "reads_after_call_count_log",
                "reads_before_call_count_log",
            ],
        },
    }


def save_feature_config(path: str = "feature_config.json") -> None:
    """Save feature configuration to JSON for reproducibility."""
    with open(path, "w") as f:
        json.dump(get_feature_config(), f, indent=2)


# ── Private encoding helpers ──────────────────────────────────────


def _one_hot(value: str, classes: list[str]) -> list[float]:
    """Encode value as one-hot vector over classes. Unknown maps to all-zeros."""
    vec = [0.0] * len(classes)
    if value in classes:
        vec[classes.index(value)] = 1.0
    return vec


def _encode_function(func: dict | None, dim: int) -> np.ndarray:
    """
    Encode a V_f node. Spec: Section 4.3 — V_f features.
    Layout: [visibility(4) | mutability(4) | is_constructor(1) | zeros...]
    """
    vec = np.zeros(dim, dtype=np.float32)
    if func is None:
        return vec

    offset = 0
    # Visibility one-hot (4)
    vis = _one_hot(func.get("visibility", ""), VISIBILITY_CLASSES)
    vec[offset:offset + len(vis)] = vis
    offset += len(vis)

    # Mutability one-hot (4)
    mut = _one_hot(func.get("mutability", ""), MUTABILITY_CLASSES)
    vec[offset:offset + len(mut)] = mut
    offset += len(mut)

    # Is constructor (1)
    vec[offset] = 1.0 if func.get("is_constructor", False) else 0.0

    return vec


def _classify_solidity_type(type_str: str) -> str:
    """Classify a Solidity type string into a category."""
    t = type_str.strip()
    tl = t.lower()

    if tl.startswith("mapping"):
        return "mapping"
    if tl.endswith("[]"):
        return "array"
    if "uint" in tl or "int" in tl:
        return "uint"
    if tl == "address":
        return "address"
    if tl == "bool":
        return "bool"
    if tl.startswith("bytes") or tl == "string":
        return "bytes"
    # Check for known struct/contract patterns (capitalized, not a primitive)
    if t[0:1].isupper() or "." in t:
        return "struct"
    return "other"


def _encode_state_var(
    var: dict | None,
    access_pattern: str,
    max_slot: int,
    dim: int,
    written_after_call: bool = False,
    read_before_call: bool = False,
) -> np.ndarray:
    """
    Encode a V_s node. Spec: Section 4.3 — V_s features (extended).
    Layout: [zeros(9)
             | type_category(8) | normalized_slot(1) | access_pattern(3)
             | written_after_call(1) | read_before_call(1)
             | zeros...]
    """
    vec = np.zeros(dim, dtype=np.float32)
    if var is None:
        return vec

    offset = N_FUNC_FEATURES  # Skip V_f feature slots

    # Type category one-hot (8)
    cat = _classify_solidity_type(var.get("type", ""))
    type_oh = _one_hot(cat, TYPE_CATEGORIES)
    vec[offset:offset + len(type_oh)] = type_oh
    offset += len(type_oh)

    # Normalized storage slot (1)
    vec[offset] = var.get("slot", 0) / max_slot
    offset += 1

    # Access pattern one-hot (3)
    ap = _one_hot(access_pattern, ACCESS_PATTERNS)
    vec[offset:offset + len(ap)] = ap
    offset += len(ap)

    # Reentrancy flags (2): is this state var ever written after an external
    # call / read before one in the same function body. These carry the core
    # reentrancy signal directly into the node features.
    vec[offset] = 1.0 if written_after_call else 0.0
    offset += 1
    vec[offset] = 1.0 if read_before_call else 0.0

    return vec


def _encode_call_site(cs: dict | None, dim: int) -> np.ndarray:
    """
    Encode a V_c node. Spec: Section 4.3 — V_c features (extended).
    Layout: [zeros(23)
             | call_opcode(4) | value_transfer(1)
             | gas_forwarded(1) | sender_controlled_target(1) | guarded_by_modifier(1)
             | writes_after_call_count_log(1) | reads_after_call_count_log(1) | reads_before_call_count_log(1)]
    """
    vec = np.zeros(dim, dtype=np.float32)
    if cs is None:
        return vec

    offset = N_FUNC_FEATURES + N_STATE_FEATURES  # Skip V_f and V_s slots

    # Call opcode one-hot (4)
    opcode = str(cs.get("opcode", "")).lower()
    if opcode not in ("call", "delegatecall", "staticcall"):
        opcode = "other"
    op_oh = _one_hot(opcode, CALL_OPCODES)
    vec[offset:offset + len(op_oh)] = op_oh
    offset += len(op_oh)

    # Value transfer flag (1)
    vec[offset] = 1.0 if cs.get("has_value", False) else 0.0
    offset += 1

    # Reentrancy flags (3): gas forwarded (exploitable), sender-controlled
    # target (attacker can trigger), guarded by a nonReentrant modifier.
    vec[offset] = 1.0 if cs.get("gas_forwarded", False) else 0.0
    offset += 1
    vec[offset] = 1.0 if cs.get("sender_controlled_target", False) else 0.0
    offset += 1
    vec[offset] = 1.0 if cs.get("guarded_by_modifier", False) else 0.0
    offset += 1

    # Reentrancy pattern counts: #state vars written-after / read-after /
    # read-before this specific call site. log1p-squashed so feature stays in
    # a small range regardless of how many vars the function touches.
    def _log_count(xs) -> float:
        return float(np.log1p(len(xs or [])))

    vec[offset] = _log_count(cs.get("writes_after_call"))
    offset += 1
    vec[offset] = _log_count(cs.get("reads_after_call"))
    offset += 1
    vec[offset] = _log_count(cs.get("reads_before_call"))

    return vec


def _compute_state_var_reentrancy_flags(
    call_sites: list[dict], state_vars: list[dict]
) -> tuple[dict[str, bool], dict[str, bool]]:
    """
    Aggregate per-call-site reentrancy context up to state variables.

    Returns:
        (written_after_call, read_before_call): two dicts keyed by state var
        name. A var is True in `written_after_call` if any call site has it in
        its writes_after_call list, and True in `read_before_call` if any call
        site has it in its reads_before_call list.
    """
    var_names = {v["name"] for v in state_vars}
    written_after: dict[str, bool] = {v: False for v in var_names}
    read_before: dict[str, bool] = {v: False for v in var_names}

    for cs in call_sites:
        for vn in cs.get("writes_after_call", []) or []:
            if vn in var_names:
                written_after[vn] = True
        for vn in cs.get("reads_before_call", []) or []:
            if vn in var_names:
                read_before[vn] = True

    return written_after, read_before


def _compute_access_patterns(cfg: dict, state_vars: list[dict]) -> dict[str, str]:
    """
    Determine access pattern (read_only / write_only / read_write) for each state var.
    Computed from CFG node-level read/write tracking across all functions.
    """
    var_names = {v["name"] for v in state_vars}
    reads = set()
    writes = set()

    for func_name, nodes in cfg.items():
        for node in nodes:
            for vn in node.get("state_vars_read", []):
                if vn in var_names:
                    reads.add(vn)
            for vn in node.get("state_vars_written", []):
                if vn in var_names:
                    writes.add(vn)

    patterns = {}
    for vn in var_names:
        is_read = vn in reads
        is_written = vn in writes
        if is_read and is_written:
            patterns[vn] = "read_write"
        elif is_read:
            patterns[vn] = "read_only"
        elif is_written:
            patterns[vn] = "write_only"
        else:
            patterns[vn] = "read_only"  # default for unused vars

    return patterns
