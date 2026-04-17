"""
Tests for Step 1 — AST, CFG, Call Graph Extraction.
Uses the withdraw example from spec Section 4.7.
"""

import os

import networkx as nx
import pytest

from src.extraction.ast_cfg import (
    build_call_graph,
    detect_pragma_version,
    extract_all,
    extract_cfg,
    extract_external_call_sites,
    extract_function_info,
    extract_slither,
    extract_state_variable_info,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
WITHDRAW_SOL = os.path.join(FIXTURES_DIR, "withdraw.sol")
NO_CALLS_SOL = os.path.join(FIXTURES_DIR, "no_calls.sol")

# Use a real contract from the dataset for integration testing
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


class TestPragmaDetection:
    def test_caret_version(self):
        assert detect_pragma_version(WITHDRAW_SOL) == "0.8.0"

    def test_caret_version_04(self):
        assert detect_pragma_version(REENTRANT_SOL) == "0.4.19"


class TestWithdrawExample:
    """Tests using the worked example from spec Section 4.7."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.result = extract_all(WITHDRAW_SOL)
        assert self.result is not None

    def test_contract_name(self):
        assert self.result["contract_name"] == "Withdraw"

    def test_ast_present(self):
        assert self.result["ast"] is not None

    def test_g_call_is_digraph(self):
        assert isinstance(self.result["G_call"], nx.DiGraph)

    def test_g_call_has_function_nodes(self):
        nodes = set(self.result["G_call"].nodes)
        assert "withdraw" in nodes
        assert "deposit" in nodes

    def test_functions_extracted(self):
        func_names = {f["name"] for f in self.result["functions"]}
        assert "withdraw" in func_names
        assert "deposit" in func_names
        # slitherConstructorVariables should be filtered out
        assert "slitherConstructorVariables" not in func_names

    def test_state_vars_extracted(self):
        var_names = {v["name"] for v in self.result["state_vars"]}
        assert "balance" in var_names

    def test_call_sites_detected(self):
        """Withdraw example should have exactly one external call site."""
        call_sites = self.result["call_sites"]
        assert len(call_sites) == 1

    def test_call_site_in_withdraw(self):
        cs = self.result["call_sites"][0]
        assert cs["function"] == "withdraw"

    def test_call_site_is_low_level_call(self):
        cs = self.result["call_sites"][0]
        assert cs["call_type"] == "low_level"
        assert str(cs["opcode"]) == "call"

    def test_call_site_has_value(self):
        cs = self.result["call_sites"][0]
        assert cs["has_value"] is True

    def test_cfg_has_withdraw(self):
        assert "withdraw" in self.result["cfg"]

    def test_cfg_withdraw_has_external_call_node(self):
        cfg_nodes = self.result["cfg"]["withdraw"]
        ext_call_nodes = [n for n in cfg_nodes if n["has_external_call"]]
        assert len(ext_call_nodes) == 1

    def test_cfg_balance_read_before_call(self):
        """balance should be read before the external call in withdraw."""
        cfg_nodes = self.result["cfg"]["withdraw"]
        call_line = None
        read_line = None
        for n in cfg_nodes:
            if n["has_external_call"]:
                call_line = n["line"]
            if "balance" in n["state_vars_read"] and call_line is None:
                read_line = n["line"]
        assert read_line is not None, "balance should be read before the call"
        assert call_line is not None, "external call should exist"
        assert read_line <= call_line

    def test_cfg_balance_written_after_call(self):
        """balance should be written after the external call in withdraw."""
        cfg_nodes = self.result["cfg"]["withdraw"]
        call_idx = None
        write_idx = None
        for i, n in enumerate(cfg_nodes):
            if n["has_external_call"]:
                call_idx = i
            if "balance" in n["state_vars_written"] and call_idx is not None:
                write_idx = i
        assert call_idx is not None
        assert write_idx is not None
        assert write_idx > call_idx

    def test_call_site_reentrancy_fields_present(self):
        """New reentrancy context fields should be populated on each call site."""
        cs = self.result["call_sites"][0]
        # Extracted via IR
        assert "gas_forwarded" in cs
        assert "sender_controlled_target" in cs
        assert "guarded_by_modifier" in cs
        # Populated by annotate_call_site_context from the CFG
        assert "writes_after_call" in cs
        assert "reads_after_call" in cs
        assert "reads_before_call" in cs

    def test_withdraw_call_is_gas_forwarding(self):
        """.call{value: ...}('') forwards all remaining gas."""
        cs = self.result["call_sites"][0]
        assert cs["gas_forwarded"] is True

    def test_withdraw_call_target_sender_controlled(self):
        """withdraw's external call is to msg.sender, attacker-controlled."""
        cs = self.result["call_sites"][0]
        assert cs["sender_controlled_target"] is True

    def test_withdraw_call_not_guarded(self):
        """withdraw has no nonReentrant modifier."""
        cs = self.result["call_sites"][0]
        assert cs["guarded_by_modifier"] is False

    def test_withdraw_balance_written_after_call(self):
        """balance is in writes_after_call for the withdraw call site."""
        cs = self.result["call_sites"][0]
        assert "balance" in cs["writes_after_call"]

    def test_withdraw_balance_read_before_call(self):
        """balance is read (via require) before the external call."""
        cs = self.result["call_sites"][0]
        assert "balance" in cs["reads_before_call"]


class TestExtractAllWithLabel:
    """When contract_label is passed, extract_all should return per-call labels."""

    def test_safe_contract_all_zero(self):
        r = extract_all(WITHDRAW_SOL, contract_label=0)
        assert r is not None
        assert "call_site_labels" in r
        assert all(v == 0 for v in r["call_site_labels"].values())
        assert r["label_info"]["contract_label"] == 0

    def test_reentrant_contract_has_labels(self):
        r = extract_all(WITHDRAW_SOL, contract_label=1)
        assert r is not None
        assert "call_site_labels" in r
        # Withdraw has one call site; Slither should flag it (or fallback does).
        assert len(r["call_site_labels"]) == 1
        assert all(v == 1 for v in r["call_site_labels"].values())


class TestNoExternalCalls:
    """Contract with no external calls should not crash."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.result = extract_all(NO_CALLS_SOL)
        assert self.result is not None

    def test_no_call_sites(self):
        assert len(self.result["call_sites"]) == 0

    def test_functions_present(self):
        func_names = {f["name"] for f in self.result["functions"]}
        assert "setValue" in func_names
        assert "getValue" in func_names

    def test_g_call_no_crash(self):
        assert isinstance(self.result["G_call"], nx.DiGraph)


class TestDatasetContract:
    """Integration test on a real dataset contract."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(REENTRANT_SOL):
            pytest.skip("Dataset not available")
        self.result = extract_all(REENTRANT_SOL, contract_name="PERSONAL_BANK")
        assert self.result is not None

    def test_contract_name(self):
        assert self.result["contract_name"] == "PERSONAL_BANK"

    def test_has_low_level_call(self):
        """PERSONAL_BANK.Collect has a low-level call (msg.sender.call.value)."""
        low_level = [cs for cs in self.result["call_sites"] if cs["call_type"] == "low_level"]
        assert len(low_level) >= 1

    def test_collect_has_value_transfer(self):
        low_level = [cs for cs in self.result["call_sites"] if cs["call_type"] == "low_level"]
        assert any(cs["has_value"] for cs in low_level)

    def test_internal_call_edge(self):
        """fallback calls Deposit."""
        assert ("fallback", "Deposit") in self.result["G_call"].edges

    def test_state_vars(self):
        var_names = {v["name"] for v in self.result["state_vars"]}
        assert "balances" in var_names
        assert "MinSum" in var_names

    def test_function_visibility(self):
        funcs = {f["name"]: f for f in self.result["functions"]}
        assert funcs["Collect"]["visibility"] == "public"
        assert funcs["Collect"]["mutability"] == "payable"
