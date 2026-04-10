"""
Tests for Step 5 — Hyperedge Construction and Incidence Matrix.
Uses the withdraw example from spec Section 4.7.
"""

import os

import networkx as nx
import numpy as np
import pytest

from src.extraction.ast_cfg import extract_all
from src.extraction.gdep import build_gdep
from src.hypergraph.hyperedges import bounded_ancestors, build_hyperedges
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


def _full_pipeline(sol_path, contract_name=None):
    r = extract_all(sol_path, contract_name=contract_name)
    assert r is not None
    G_dep = build_gdep(r["cfg"], r["call_sites"], r["state_vars"])
    ns = build_node_sets(r["functions"], r["state_vars"], r["call_sites"])
    E, H_inc = build_hyperedges(
        ns["V"], ns["V_c"], ns["V_s"], ns["node_index"],
        r["G_call"], G_dep, r["call_sites"],
    )
    return r, G_dep, ns, E, H_inc


class TestBoundedAncestors:
    def test_simple_chain(self):
        """A -> B -> C -> D: ancestors of D with delta=2 should be {C, B}."""
        G = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "D")])
        assert bounded_ancestors(G, "D", delta=2) == {"C", "B"}

    def test_depth_bound(self):
        """A -> B -> C -> D: ancestors of D with delta=1 should be {C} only."""
        G = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "D")])
        assert bounded_ancestors(G, "D", delta=1) == {"C"}

    def test_full_depth(self):
        """A -> B -> C -> D: ancestors of D with delta=3 should be {C, B, A}."""
        G = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "D")])
        assert bounded_ancestors(G, "D", delta=3) == {"A", "B", "C"}

    def test_no_ancestors(self):
        """Root node has no ancestors."""
        G = nx.DiGraph([("A", "B")])
        assert bounded_ancestors(G, "A", delta=3) == set()

    def test_node_not_in_graph(self):
        G = nx.DiGraph([("A", "B")])
        assert bounded_ancestors(G, "Z", delta=3) == set()

    def test_diamond(self):
        """A -> C, B -> C: ancestors of C with delta=1 should be {A, B}."""
        G = nx.DiGraph([("A", "C"), ("B", "C")])
        assert bounded_ancestors(G, "C", delta=1) == {"A", "B"}


class TestWithdrawHyperedges:
    """Hyperedge tests using the worked example from spec Section 4.7."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r, self.G_dep, self.ns, self.E, self.H_inc = _full_pipeline(WITHDRAW_SOL)

    def test_one_hyperedge_per_call_site(self):
        assert len(self.E) == len(self.ns["V_c"])

    def test_hyperedge_count_equals_one(self):
        """Withdraw has exactly one external call site."""
        assert len(self.E) == 1

    def test_hyperedge_contains_withdraw(self):
        """Spec Section 4.7: e_c should contain withdraw."""
        assert "func:withdraw" in self.E[0]

    def test_hyperedge_contains_balance(self):
        """Spec Section 4.7: e_c should contain balance."""
        assert "var:balance" in self.E[0]

    def test_hyperedge_contains_call_site(self):
        """Spec Section 4.7: e_c should contain the call site node."""
        cs = self.ns["V_c"][0]
        assert cs in self.E[0]

    def test_hyperedge_has_three_members(self):
        """Spec Section 4.7: e_c = {withdraw, balance, c} — 3 members."""
        assert len(self.E[0]) == 3

    def test_h_inc_shape(self):
        assert self.H_inc.shape == (len(self.ns["V"]), len(self.ns["V_c"]))

    def test_h_inc_binary(self):
        assert set(self.H_inc.flatten().tolist()) <= {0.0, 1.0}

    def test_h_inc_call_site_in_own_hyperedge(self):
        cs = self.ns["V_c"][0]
        idx = self.ns["node_index"][cs]
        assert self.H_inc[idx, 0] == 1.0

    def test_h_inc_deposit_not_in_hyperedge(self):
        """deposit is not called by withdraw, so should not be in the hyperedge."""
        idx = self.ns["node_index"]["func:deposit"]
        assert self.H_inc[idx, 0] == 0.0


class TestNoCallsHyperedges:
    """Contract with no external calls should produce empty E and H_inc."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r, self.G_dep, self.ns, self.E, self.H_inc = _full_pipeline(NO_CALLS_SOL)

    def test_no_hyperedges(self):
        assert len(self.E) == 0

    def test_h_inc_shape(self):
        assert self.H_inc.shape == (len(self.ns["V"]), 0)


class TestDatasetHyperedges:
    """Integration test on PERSONAL_BANK."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(REENTRANT_SOL):
            pytest.skip("Dataset not available")
        self.r, self.G_dep, self.ns, self.E, self.H_inc = _full_pipeline(
            REENTRANT_SOL, "PERSONAL_BANK"
        )

    def test_three_hyperedges(self):
        """PERSONAL_BANK has 3 external call sites."""
        assert len(self.E) == 3

    def test_collect_47_contains_balances(self):
        """The reentrancy call site should include var:balances."""
        # Find the hyperedge for call:Collect:47
        for e in self.E:
            if "call:Collect:47" in e:
                assert "var:balances" in e
                return
        pytest.fail("call:Collect:47 not found in any hyperedge")

    def test_deposit_hyperedge_has_ancestor(self):
        """Deposit's hyperedge should include fallback (ancestor via G_call)."""
        for e in self.E:
            if "call:Deposit:38" in e:
                assert "func:fallback" in e, "fallback should be ancestor of Deposit"
                return
        pytest.fail("call:Deposit:38 not found in any hyperedge")

    def test_h_inc_shape(self):
        assert self.H_inc.shape == (len(self.ns["V"]), 3)

    def test_h_inc_binary(self):
        assert set(self.H_inc.flatten().tolist()) <= {0.0, 1.0}

    def test_every_call_site_in_own_hyperedge(self):
        for j, c in enumerate(self.ns["V_c"]):
            idx = self.ns["node_index"][c]
            assert self.H_inc[idx, j] == 1.0, f"{c} not in its own hyperedge column {j}"
