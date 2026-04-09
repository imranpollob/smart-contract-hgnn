"""
Tests for Step 3 — Node Set Construction.
Uses the withdraw example from spec Section 4.7.
"""

import os

import pytest

from src.extraction.ast_cfg import extract_all
from src.hypergraph.nodeset import build_node_sets

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
WITHDRAW_SOL = os.path.join(FIXTURES_DIR, "withdraw.sol")
NO_CALLS_SOL = os.path.join(FIXTURES_DIR, "no_calls.sol")

DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "reentrancy-detection-benchmarks",
    "benchmarks",
    "aggregated-benchmark",
    "src",
)
REENTRANT_SOL = os.path.join(
    DATASET_DIR, "reentrant", "0cdc64278f3169e4b4be494f50a9067f_cgt.sol"
)


class TestWithdrawNodeSet:
    """Node set tests using the withdraw example from spec Section 4.7."""

    @pytest.fixture(autouse=True)
    def setup(self):
        r = extract_all(WITHDRAW_SOL)
        assert r is not None
        self.ns = build_node_sets(r["functions"], r["state_vars"], r["call_sites"])

    def test_v_f_contains_functions(self):
        assert "func:deposit" in self.ns["V_f"]
        assert "func:withdraw" in self.ns["V_f"]

    def test_v_s_contains_balance(self):
        assert "var:balance" in self.ns["V_s"]

    def test_v_c_has_one_call_site(self):
        """Withdraw example: V_c should contain exactly one call site node."""
        assert len(self.ns["V_c"]) == 1
        assert self.ns["V_c"][0].startswith("call:withdraw:")

    def test_disjointness(self):
        set_f = set(self.ns["V_f"])
        set_s = set(self.ns["V_s"])
        set_c = set(self.ns["V_c"])
        assert set_f.isdisjoint(set_s)
        assert set_f.isdisjoint(set_c)
        assert set_s.isdisjoint(set_c)

    def test_v_length(self):
        assert len(self.ns["V"]) == len(self.ns["V_f"]) + len(self.ns["V_s"]) + len(self.ns["V_c"])

    def test_node_index_covers_all(self):
        assert len(self.ns["node_index"]) == len(self.ns["V"])
        for node in self.ns["V"]:
            assert node in self.ns["node_index"]

    def test_node_index_is_sequential(self):
        indices = sorted(self.ns["node_index"].values())
        assert indices == list(range(len(self.ns["V"])))

    def test_v_ordering(self):
        """V should be ordered: V_f first, then V_s, then V_c."""
        v = self.ns["V"]
        n_f = len(self.ns["V_f"])
        n_s = len(self.ns["V_s"])
        for i in range(n_f):
            assert v[i].startswith("func:")
        for i in range(n_f, n_f + n_s):
            assert v[i].startswith("var:")
        for i in range(n_f + n_s, len(v)):
            assert v[i].startswith("call:")


class TestNoCallsNodeSet:
    """Contract with no external calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        r = extract_all(NO_CALLS_SOL)
        assert r is not None
        self.ns = build_node_sets(r["functions"], r["state_vars"], r["call_sites"])

    def test_empty_v_c(self):
        assert len(self.ns["V_c"]) == 0

    def test_v_f_present(self):
        assert len(self.ns["V_f"]) >= 2

    def test_v_s_present(self):
        assert len(self.ns["V_s"]) >= 1


class TestDatasetNodeSet:
    """Integration test on PERSONAL_BANK."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(REENTRANT_SOL):
            pytest.skip("Dataset not available")
        r = extract_all(REENTRANT_SOL, contract_name="PERSONAL_BANK")
        assert r is not None
        self.ns = build_node_sets(r["functions"], r["state_vars"], r["call_sites"])

    def test_v_f_has_collect(self):
        assert "func:Collect" in self.ns["V_f"]

    def test_v_s_has_balances(self):
        assert "var:balances" in self.ns["V_s"]

    def test_v_c_has_call_sites(self):
        """PERSONAL_BANK has 3 external call sites."""
        assert len(self.ns["V_c"]) == 3

    def test_disjointness(self):
        set_f = set(self.ns["V_f"])
        set_s = set(self.ns["V_s"])
        set_c = set(self.ns["V_c"])
        assert set_f.isdisjoint(set_s)
        assert set_f.isdisjoint(set_c)
        assert set_s.isdisjoint(set_c)
