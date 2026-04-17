"""
Per-call-site labeling via Slither reentrancy detectors.
Spec: Section 6.2 (revised) — Hyperedge-level Label Assignment.

For each external call site c in a contract, determine whether that specific
call site is flagged as reentrant by Slither's reentrancy detectors. This
replaces the original directory-based contract-level labeling, which smeared
a single label onto every call site in a contract regardless of which call was
actually vulnerable.
"""

import logging

logger = logging.getLogger(__name__)

# Slither ships several reentrancy detectors. We import them defensively so a
# version mismatch in one does not disable the rest.
_DETECTOR_CLASSES = []
try:
    from slither.detectors.reentrancy.reentrancy_eth import ReentrancyEth
    _DETECTOR_CLASSES.append(ReentrancyEth)
except Exception as e:
    logger.debug("ReentrancyEth unavailable: %s", e)

try:
    from slither.detectors.reentrancy.reentrancy_read_before_write import (
        ReentrancyReadBeforeWritten,
    )
    _DETECTOR_CLASSES.append(ReentrancyReadBeforeWritten)
except Exception as e:
    logger.debug("ReentrancyReadBeforeWritten unavailable: %s", e)

try:
    from slither.detectors.reentrancy.reentrancy_no_gas import ReentrancyNoGas
    _DETECTOR_CLASSES.append(ReentrancyNoGas)
except Exception as e:
    logger.debug("ReentrancyNoGas unavailable: %s", e)

try:
    from slither.detectors.reentrancy.reentrancy_benign import ReentrancyBenign
    _DETECTOR_CLASSES.append(ReentrancyBenign)
except Exception as e:
    logger.debug("ReentrancyBenign unavailable: %s", e)

try:
    from slither.detectors.reentrancy.reentrancy_events import ReentrancyEvent
    _DETECTOR_CLASSES.append(ReentrancyEvent)
except Exception as e:
    logger.debug("ReentrancyEvent unavailable: %s", e)


def label_call_sites(
    slither_obj,
    contract_name: str,
    call_sites: list[dict],
    contract_label: int,
) -> dict:
    """
    Assign a per-call-site label (0 or 1) to every call site in `call_sites`.
    Spec: Section 6.2 (revised) — per-hyperedge label assignment.

    Policy:
        - Safe contracts (contract_label=0): every call site gets 0.
        - Reentrant contracts (contract_label=1): a call site gets 1 only if it
          is flagged by at least one Slither reentrancy detector. If Slither
          flags no call site in a reentrant contract (detector miss), we fall
          back to the old contract-level policy and label all call sites 1 —
          losing a true positive is worse than slight noise.

    Args:
        slither_obj: Slither analysis result for the contract
        contract_name: the resolved target contract name
        call_sites: list of call site dicts from extract_external_call_sites
        contract_label: directory-derived label (0=safe, 1=reentrant)

    Returns:
        dict with keys:
          - 'labels': dict mapping "call:<function>:<line>" -> 0/1
          - 'flagged_by_slither': set of flagged call-site ids
          - 'fallback_used': True if contract_label=1 but Slither flagged nothing
    """
    cs_ids = [f"call:{cs['function']}:{cs['line']}" for cs in call_sites]

    # Safe contracts: everything is 0.
    if contract_label == 0:
        return {
            "labels": {cid: 0 for cid in cs_ids},
            "flagged_by_slither": set(),
            "fallback_used": False,
        }

    # Reentrant contract: ask Slither which call sites are actually flagged.
    flagged = _run_detectors(slither_obj, contract_name, call_sites)

    if flagged:
        return {
            "labels": {cid: (1 if cid in flagged else 0) for cid in cs_ids},
            "flagged_by_slither": flagged,
            "fallback_used": False,
        }

    # Detector miss on a reentrant contract: fall back to contract-level label.
    logger.debug(
        "Slither flagged no call sites for reentrant contract %s; using fallback",
        contract_name,
    )
    return {
        "labels": {cid: 1 for cid in cs_ids},
        "flagged_by_slither": set(),
        "fallback_used": True,
    }


def _run_detectors(slither_obj, contract_name: str, call_sites: list[dict]) -> set[str]:
    """
    Run the registered reentrancy detectors and return the set of call-site
    IDs (call:<function>:<line>) that match a detector finding.

    We match a finding element against our known call sites by (function_name,
    line). This is robust because `call_sites` only contains actual external
    calls, so incidental matches (e.g. the state-write node that also appears
    in the finding's elements list) are naturally filtered out.
    """
    if not _DETECTOR_CLASSES:
        return set()

    # Build lookup: (function_name, line) -> call_id
    cs_lookup: dict[tuple[str, int], str] = {}
    for cs in call_sites:
        cs_lookup[(cs["function"], cs["line"])] = f"call:{cs['function']}:{cs['line']}"

    try:
        for DC in _DETECTOR_CLASSES:
            slither_obj.register_detector(DC)
        results_list = slither_obj.run_detectors()
    except Exception as e:
        logger.warning("Slither detector run failed for %s: %s", contract_name, e)
        return set()

    flagged: set[str] = set()
    for results in results_list:
        if not results:
            continue
        for finding in results:
            for el in finding.get("elements", []):
                if el.get("type") != "node":
                    continue
                ts = el.get("type_specific_fields", {})
                parent = ts.get("parent", {})
                if parent.get("type") != "function":
                    continue
                grand = parent.get("type_specific_fields", {}).get("parent", {})
                if grand.get("name") and grand.get("name") != contract_name:
                    continue

                func_name = parent.get("name")
                lines = el.get("source_mapping", {}).get("lines") or []
                if not func_name or not lines:
                    continue

                for line in lines:
                    key = (func_name, line)
                    if key in cs_lookup:
                        flagged.add(cs_lookup[key])
                        break

    return flagged
