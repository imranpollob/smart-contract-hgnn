"""
Tests for Step 2 — Data Dependency Graph (G_dep).
Uses the withdraw example from spec Section 4.7.
"""

import os

import networkx as nx
import pytest

from src.extraction.ast_cfg import extract_all
from src.extraction.gdep import build_gdep

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


class TestWithdrawGdep:
    """G_dep tests using the worked example from spec Section 4.7."""

    @pytest.fixture(autouse=True)
    def setup(self):
        r = extract_all(WITHDRAW_SOL)
        assert r is not None
        self.G_dep = build_gdep(r["cfg"], r["call_sites"], r["state_vars"])

    def test_is_digraph(self):
        assert isinstance(self.G_dep, nx.DiGraph)

    def test_call_site_node_exists(self):
        assert "call:withdraw:13" in self.G_dep

    def test_balance_var_node_exists(self):
        assert "var:balance" in self.G_dep

    def test_dependency_edge(self):
        """Spec Section 4.7: (external_call, balance[msg.sender]) must be in G_dep."""
        assert ("call:withdraw:13", "var:balance") in self.G_dep.edges

    def test_bipartite_structure(self):
        """No state variable should appear as a left-side (call site) node."""
        for u, v in self.G_dep.edges:
            assert u.startswith("call:"), f"Left node should be call site, got {u}"
            assert v.startswith("var:"), f"Right node should be state var, got {v}"

    def test_single_call_site(self):
        """Withdraw has exactly one external call site."""
        call_nodes = [n for n in self.G_dep.nodes if n.startswith("call:")]
        assert len(call_nodes) == 1


class TestNoCallsGdep:
    """Contract with no external calls should produce G_dep with no edges."""

    @pytest.fixture(autouse=True)
    def setup(self):
        r = extract_all(NO_CALLS_SOL)
        assert r is not None
        self.G_dep = build_gdep(r["cfg"], r["call_sites"], r["state_vars"])

    def test_no_edges(self):
        assert len(self.G_dep.edges) == 0

    def test_no_call_site_nodes(self):
        call_nodes = [n for n in self.G_dep.nodes if n.startswith("call:")]
        assert len(call_nodes) == 0

    def test_state_var_nodes_present(self):
        """State vars should still be added as nodes even with no call sites."""
        var_nodes = [n for n in self.G_dep.nodes if n.startswith("var:")]
        assert len(var_nodes) >= 1


class TestDatasetGdep:
    """Integration test on PERSONAL_BANK contract."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(REENTRANT_SOL):
            pytest.skip("Dataset not available")
        r = extract_all(REENTRANT_SOL, contract_name="PERSONAL_BANK")
        assert r is not None
        self.G_dep = build_gdep(r["cfg"], r["call_sites"], r["state_vars"])

    def test_collect_low_level_call_depends_on_balances(self):
        """Collect's msg.sender.call.value (line 47) should depend on balances."""
        assert ("call:Collect:47", "var:balances") in self.G_dep.edges

    def test_collect_low_level_call_depends_on_minsum(self):
        """Collect reads MinSum before the call."""
        assert ("call:Collect:47", "var:MinSum") in self.G_dep.edges

    def test_bipartite_structure(self):
        for u, v in self.G_dep.edges:
            assert u.startswith("call:")
            assert v.startswith("var:")
