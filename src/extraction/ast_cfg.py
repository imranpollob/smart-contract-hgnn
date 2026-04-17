"""
Step 1 — AST, CFG, Call Graph Extraction.
Spec: Section 3.2 (Program Representation Layer).

Parses a Solidity contract and extracts:
- AST via solc --ast-compact-json
- CFG via Slither (per-function control flow graph nodes)
- Call graph G_call as nx.DiGraph
"""

import json
import logging
import os
import re
import subprocess

import networkx as nx
from slither.slither import Slither
from slither.slithir.operations import (
    HighLevelCall,
    InternalCall,
    LowLevelCall,
)

logger = logging.getLogger(__name__)

# Modifier names commonly used to guard against reentrancy. Matching is
# case-insensitive substring so variants like `nonReentrant`, `noReentrancy`,
# `preventReentrant`, `mutex` are caught.
_REENTRANCY_GUARD_KEYWORDS = (
    "nonreentrant",
    "noreentrant",
    "noreentrancy",
    "preventreentrant",
    "preventrecursion",
    "mutex",
    "lock",
)

# Low-level opcodes that forward all remaining gas (and are therefore the ones
# exploitable by a reentrancy attacker). `send`/`transfer` forward only the
# 2300-gas stipend so they cannot re-enter.
_GAS_FORWARDING_OPCODES = {"call", "delegatecall", "staticcall"}


def detect_pragma_version(sol_path: str) -> str | None:
    """
    Parse the pragma solidity line and return a concrete compiler version to use.
    Handles formats: ^0.4.19, >=0.4.22 <0.6.0, 0.4.25, etc.
    Returns the minimum satisfying version as a concrete version string.
    """
    with open(sol_path, "r") as f:
        content = f.read(2000)

    match = re.search(r"pragma\s+solidity\s+([^;]+)", content)
    if not match:
        return None

    pragma = match.group(1).strip()

    # Extract all version numbers mentioned
    versions = re.findall(r"(\d+\.\d+\.\d+)", pragma)
    if not versions:
        return None

    # Use the first (minimum) version mentioned
    return versions[0]


def install_and_use_solc(version: str) -> bool:
    """Install a solc version via solc-select and set it as active."""
    try:
        subprocess.run(
            ["solc-select", "install", version],
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = subprocess.run(
            ["solc-select", "use", version],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Failed to install/use solc %s: %s", version, e)
        return False


def extract_ast(sol_path: str, solc_version: str | None = None) -> dict | None:
    """
    Extract AST from a Solidity file using solc --ast-compact-json.
    Spec: Section 3.2 — AST extraction.

    Returns the parsed AST dict, or None on failure.
    """
    if solc_version and not install_and_use_solc(solc_version):
        logger.warning("Could not set solc version %s for %s", solc_version, sol_path)

    try:
        result = subprocess.run(
            ["solc", "--ast-compact-json", sol_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning("solc failed on %s: %s", sol_path, result.stderr[:500])
            return None

        # solc outputs a header line then JSON; find the JSON start
        output = result.stdout
        json_start = output.find("{")
        if json_start == -1:
            logger.warning("No JSON found in solc output for %s", sol_path)
            return None

        return json.loads(output[json_start:])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning("AST extraction failed for %s: %s", sol_path, e)
        return None


def extract_slither(sol_path: str) -> Slither | None:
    """
    Run Slither on a Solidity file.
    Returns the Slither object or None on failure.
    """
    try:
        return Slither(sol_path)
    except Exception as e:
        logger.warning("Slither failed on %s: %s", sol_path, e)
        return None


def build_call_graph(slither_obj: Slither, contract_name: str | None = None) -> nx.DiGraph:
    """
    Build call graph G_call from Slither analysis.
    Spec: Section 3.2 — Call Graph G_call.

    G_call is a directed graph where edge f_i → f_j means f_i calls f_j.
    Nodes are function names prefixed with contract name for uniqueness.

    Args:
        slither_obj: Slither analysis result
        contract_name: If provided, only extract from this contract.
                       If None, use the first non-library contract.

    Returns:
        G_call: nx.DiGraph with function nodes and call edges
    """
    G_call = nx.DiGraph()

    target_contract = _select_contract(slither_obj, contract_name)
    if target_contract is None:
        return G_call

    # Add all function nodes (skip synthetic constructor-variable initializer)
    for func in target_contract.functions:
        if func.name == "slitherConstructorVariables":
            continue
        G_call.add_node(func.name)

    # Add call edges from IR-level InternalCall operations
    for func in target_contract.functions:
        if func.name == "slitherConstructorVariables":
            continue
        for node in func.nodes:
            for ir in node.irs:
                if isinstance(ir, InternalCall):
                    called_func = getattr(ir, "function", None)
                    if called_func and hasattr(called_func, "name"):
                        called_name = called_func.name
                        if called_name in G_call and called_name != func.name:
                            G_call.add_edge(func.name, called_name)

    return G_call


def extract_external_call_sites(
    slither_obj: Slither, contract_name: str | None = None
) -> list[dict]:
    """
    Identify all external call sites (V_c candidates) from a contract.
    Spec: Section 4.2 — V_c = external call site nodes.

    Each call site dict contains the base fields (function, line, call_type,
    opcode, has_value) plus reentrancy-specific fields derived from the CFG
    and IR (Section 4.3, extended):
        - gas_forwarded: True for .call/.delegatecall/.staticcall (which forward
          all remaining gas and are therefore exploitable); False for .send /
          .transfer (which forward only the 2300-gas stipend)
        - sender_controlled_target: True if the call target is msg.sender or a
          parameter of type address / address payable
        - guarded_by_modifier: True if the enclosing function carries a
          nonReentrant-style modifier
        - writes_after_call: list of state variable names written after the
          call within the same function body
        - reads_after_call: list of state variable names read after the call
        - reads_before_call: list of state variable names read before the call

    Returns:
        List of call site dicts
    """
    target_contract = _select_contract(slither_obj, contract_name)
    if target_contract is None:
        return []

    call_sites = []

    for func in target_contract.functions:
        if func.name == "slitherConstructorVariables":
            continue

        guarded = _has_reentrancy_guard(func)

        for node in func.nodes:
            # Low-level calls: .call(), .delegatecall(), .staticcall(), .send(), .transfer()
            for ir in node.irs:
                if isinstance(ir, LowLevelCall):
                    line = node.source_mapping.lines[0] if node.source_mapping.lines else 0
                    opcode = ir.function_name if hasattr(ir, "function_name") else "call"
                    call_sites.append({
                        "function": func.name,
                        "line": line,
                        "call_type": "low_level",
                        "opcode": opcode,
                        "has_value": _ir_has_value(ir),
                        "gas_forwarded": str(opcode).lower() in _GAS_FORWARDING_OPCODES,
                        "sender_controlled_target": _is_sender_controlled_target(ir, func, node),
                        "guarded_by_modifier": guarded,
                    })
                elif isinstance(ir, HighLevelCall):
                    # Only count calls to external contracts, not internal
                    dest = ir.destination
                    # destination can be a Variable (with .type) or a Contract directly
                    if hasattr(dest, "type"):
                        dest_name = str(dest.type)
                    else:
                        dest_name = str(dest)
                    if dest_name != target_contract.name:
                        line = node.source_mapping.lines[0] if node.source_mapping.lines else 0
                        opcode = ir.function_name if hasattr(ir, "function_name") else "unknown"
                        call_sites.append({
                            "function": func.name,
                            "line": line,
                            "call_type": "high_level",
                            "opcode": opcode,
                            "has_value": _ir_has_value(ir),
                            # High-level calls forward gas by default unless `.gas(0)` is used.
                            "gas_forwarded": True,
                            "sender_controlled_target": _is_sender_controlled_target(ir, func, node),
                            "guarded_by_modifier": guarded,
                        })

    return call_sites


def extract_state_variable_info(
    slither_obj: Slither, contract_name: str | None = None
) -> list[dict]:
    """
    Extract state variable metadata from a contract.
    Spec: Section 4.2 — V_s = state variable nodes.

    Returns list of dicts with:
        - name: variable name
        - type: Solidity type string
        - slot: storage slot index (order of declaration)
    """
    target_contract = _select_contract(slither_obj, contract_name)
    if target_contract is None:
        return []

    state_vars = []
    for i, var in enumerate(target_contract.state_variables):
        state_vars.append({
            "name": var.name,
            "type": str(var.type),
            "slot": i,
        })
    return state_vars


def extract_function_info(
    slither_obj: Slither, contract_name: str | None = None
) -> list[dict]:
    """
    Extract function metadata from a contract.
    Spec: Section 4.2 — V_f = function nodes.

    Returns list of dicts with:
        - name: function name
        - visibility: public/private/internal/external
        - mutability: pure/view/payable/nonpayable
        - is_constructor: bool
    """
    target_contract = _select_contract(slither_obj, contract_name)
    if target_contract is None:
        return []

    functions = []
    for func in target_contract.functions:
        if func.name == "slitherConstructorVariables":
            continue

        if func.pure:
            mutability = "pure"
        elif func.view:
            mutability = "view"
        elif func.payable:
            mutability = "payable"
        else:
            mutability = "nonpayable"

        functions.append({
            "name": func.name,
            "visibility": func.visibility,
            "mutability": mutability,
            "is_constructor": func.is_constructor,
        })
    return functions


def extract_cfg(slither_obj: Slither, contract_name: str | None = None) -> dict:
    """
    Extract CFG per function from Slither.
    Spec: Section 3.2 — CFG extraction.

    Returns dict mapping function_name -> list of CFG node dicts.
    Each CFG node dict has:
        - type: node type string
        - expression: string representation
        - line: source line number
        - state_vars_read: list of state variable names read
        - state_vars_written: list of state variable names written
        - has_external_call: bool
    """
    target_contract = _select_contract(slither_obj, contract_name)
    if target_contract is None:
        return {}

    cfg = {}
    for func in target_contract.functions:
        if func.name == "slitherConstructorVariables":
            continue

        nodes = []
        for node in func.nodes:
            lines = node.source_mapping.lines if node.source_mapping.lines else []
            has_ext_call = bool(node.low_level_calls or node.high_level_calls)

            nodes.append({
                "type": str(node.type),
                "expression": str(node) if node.expression else "",
                "line": lines[0] if lines else 0,
                "state_vars_read": [v.name for v in node.state_variables_read],
                "state_vars_written": [v.name for v in node.state_variables_written],
                "has_external_call": has_ext_call,
            })
        cfg[func.name] = nodes

    return cfg


def extract_all(
    sol_path: str,
    contract_name: str | None = None,
    contract_label: int | None = None,
) -> dict | None:
    """
    Run the full extraction pipeline for a single Solidity file.
    Spec: Section 3.2 — Program Representation Layer.

    Args:
        sol_path: path to the .sol file
        contract_name: if set, restrict to this contract; otherwise a heuristic picks one
        contract_label: directory-derived label (0=safe, 1=reentrant). When
            provided, per-call-site labels are computed via Slither's
            reentrancy detectors (see src/extraction/labels.py). When None,
            labeling is skipped and callers can apply their own policy.

    Returns dict with keys:
        ast, G_call, cfg, call_sites, state_vars, functions, contract_name,
        sol_path, and (if contract_label is not None) call_site_labels +
        label_info.

    Returns None if Slither fails on the file.
    """
    # Detect and set solc version
    version = detect_pragma_version(sol_path)
    if version:
        install_and_use_solc(version)

    # Extract AST (optional — may fail for some contracts)
    ast = extract_ast(sol_path, version)

    # Run Slither
    slither_obj = extract_slither(sol_path)
    if slither_obj is None:
        return None

    # Select the target contract
    target = _select_contract(slither_obj, contract_name)
    if target is None:
        return None

    resolved_name = target.name

    cfg = extract_cfg(slither_obj, resolved_name)
    call_sites = extract_external_call_sites(slither_obj, resolved_name)
    # Annotate each call site with reads_before / reads_after / writes_after
    # computed from the CFG ordering.
    annotate_call_site_context(call_sites, cfg)

    result = {
        "ast": ast,
        "G_call": build_call_graph(slither_obj, resolved_name),
        "cfg": cfg,
        "call_sites": call_sites,
        "state_vars": extract_state_variable_info(slither_obj, resolved_name),
        "functions": extract_function_info(slither_obj, resolved_name),
        "contract_name": resolved_name,
        "sol_path": sol_path,
    }

    if contract_label is not None:
        # Import lazily so labels.py failing to import doesn't break extraction.
        from src.extraction.labels import label_call_sites

        label_info = label_call_sites(
            slither_obj, resolved_name, call_sites, contract_label
        )
        result["call_site_labels"] = label_info["labels"]
        result["label_info"] = {
            "contract_label": contract_label,
            "flagged_by_slither": sorted(label_info["flagged_by_slither"]),
            "fallback_used": label_info["fallback_used"],
        }

    return result


# ── Private helpers ──────────────────────────────────────────────


def _select_contract(slither_obj: Slither, contract_name: str | None) -> object | None:
    """Select target contract from Slither results."""
    if contract_name:
        for c in slither_obj.contracts:
            if c.name == contract_name:
                return c
        logger.warning("Contract %s not found", contract_name)
        return None

    # Heuristic: pick the first contract that has functions and isn't an interface/library
    for c in slither_obj.contracts:
        if c.is_library or c.is_interface:
            continue
        if len(c.functions) > 0:
            return c
    # Fallback: first contract
    return slither_obj.contracts[0] if slither_obj.contracts else None


def _ir_has_value(ir) -> bool:
    """Check if an IR call operation transfers ETH value."""
    if hasattr(ir, "call_value") and ir.call_value is not None:
        return True
    return False


def _has_reentrancy_guard(func) -> bool:
    """True if the function carries a modifier whose name looks like a guard."""
    modifiers = getattr(func, "modifiers", None) or []
    for m in modifiers:
        name = str(getattr(m, "name", "")).lower()
        if any(kw in name for kw in _REENTRANCY_GUARD_KEYWORDS):
            return True
    return False


def _is_sender_controlled_target(ir, func, node) -> bool:
    """
    Heuristic: True if the call destination is attacker-influenceable.

    Cheap checks:
    - expression text at the call node contains `msg.sender.`
    - IR destination is a function parameter (any caller can supply the address)
    """
    expr = str(node) if getattr(node, "expression", None) else ""
    if "msg.sender." in expr or "msg.sender.call" in expr:
        return True

    dest = getattr(ir, "destination", None)
    if dest is None:
        return False

    # If destination is one of the function's declared parameters, a caller
    # controls it.
    try:
        param_names = {getattr(p, "name", None) for p in getattr(func, "parameters", [])}
        if getattr(dest, "name", None) in param_names and param_names:
            return True
    except Exception:
        pass

    return False


def annotate_call_site_context(
    call_sites: list[dict], cfg: dict
) -> list[dict]:
    """
    Populate `writes_after_call`, `reads_after_call`, `reads_before_call` on
    each call site by walking the enclosing function's CFG.

    Modifies call_sites in-place and returns the same list.
    """
    for cs in call_sites:
        func_name = cs["function"]
        call_line = cs["line"]
        nodes = cfg.get(func_name, [])

        call_idx = None
        for i, n in enumerate(nodes):
            if n.get("has_external_call") and n.get("line") == call_line:
                call_idx = i
                break
        if call_idx is None:
            for i, n in enumerate(nodes):
                if n.get("has_external_call"):
                    call_idx = i
                    break

        reads_before: set[str] = set()
        reads_after: set[str] = set()
        writes_after: set[str] = set()

        if call_idx is not None:
            for i in range(call_idx):
                reads_before.update(nodes[i].get("state_vars_read", []))
            for i in range(call_idx + 1, len(nodes)):
                reads_after.update(nodes[i].get("state_vars_read", []))
                writes_after.update(nodes[i].get("state_vars_written", []))

        cs["reads_before_call"] = sorted(reads_before)
        cs["reads_after_call"] = sorted(reads_after)
        cs["writes_after_call"] = sorted(writes_after)

    return call_sites
