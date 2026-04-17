"""
Tests for Section 6.2 (revised) — per-call-site reentrancy labeling.
"""

import os

import pytest

from src.extraction.ast_cfg import (
    detect_pragma_version,
    extract_all,
    extract_external_call_sites,
    extract_slither,
    install_and_use_solc,
)
from src.extraction.labels import label_call_sites

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
WITHDRAW_SOL = os.path.join(FIXTURES_DIR, "withdraw.sol")
NO_CALLS_SOL = os.path.join(FIXTURES_DIR, "no_calls.sol")


def _slither_for(sol_path):
    """Ensure the right solc is active before invoking Slither directly."""
    version = detect_pragma_version(sol_path)
    if version:
        install_and_use_solc(version)
    return extract_slither(sol_path)


class TestSafeContractLabeling:
    """Safe contracts always get all-zero labels regardless of Slither."""

    def test_all_labels_zero(self):
        slither_obj = _slither_for(WITHDRAW_SOL)
        assert slither_obj is not None
        call_sites = extract_external_call_sites(slither_obj)
        result = label_call_sites(slither_obj, "Withdraw", call_sites, contract_label=0)
        assert all(v == 0 for v in result["labels"].values())
        assert result["fallback_used"] is False


class TestReentrantContractLabeling:
    """Reentrant contracts get per-site labels from Slither detectors."""

    def test_withdraw_flags_call_site(self):
        slither_obj = _slither_for(WITHDRAW_SOL)
        assert slither_obj is not None
        call_sites = extract_external_call_sites(slither_obj)
        result = label_call_sites(slither_obj, "Withdraw", call_sites, contract_label=1)
        # All existing call sites should end up labeled 1 (either flagged or
        # reached via the fallback path).
        assert len(result["labels"]) == len(call_sites)
        assert all(v == 1 for v in result["labels"].values())

    def test_labels_are_binary(self):
        slither_obj = _slither_for(WITHDRAW_SOL)
        call_sites = extract_external_call_sites(slither_obj)
        result = label_call_sites(slither_obj, "Withdraw", call_sites, contract_label=1)
        for v in result["labels"].values():
            assert v in (0, 1)


class TestFallbackSemantics:
    """When no detector flags anything, we fall back to contract-level label."""

    def test_empty_call_sites_returns_empty_labels(self):
        slither_obj = _slither_for(NO_CALLS_SOL)
        assert slither_obj is not None
        result = label_call_sites(slither_obj, "NoCalls", [], contract_label=1)
        assert result["labels"] == {}


class TestExtractAllIntegration:
    """extract_all should thread contract_label through to labels."""

    def test_label_info_populated(self):
        r = extract_all(WITHDRAW_SOL, contract_label=1)
        assert r is not None
        assert "label_info" in r
        assert r["label_info"]["contract_label"] == 1
        assert "flagged_by_slither" in r["label_info"]
        assert "fallback_used" in r["label_info"]
