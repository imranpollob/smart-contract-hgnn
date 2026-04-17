"""
Tests for Step 4 — Node Feature Matrix (X).
Uses the withdraw example from spec Section 4.7.
"""

import os

import numpy as np
import pytest

from src.extraction.ast_cfg import extract_all
from src.hypergraph.features import (
    FEATURE_DIM,
    N_CALL_FEATURES,
    N_FUNC_FEATURES,
    N_STATE_FEATURES,
    build_feature_matrix,
    get_feature_config,
    _classify_solidity_type,
    _compute_state_var_reentrancy_flags,
)
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


def _extract_and_build(sol_path, contract_name=None):
    r = extract_all(sol_path, contract_name=contract_name)
    assert r is not None
    ns = build_node_sets(r["functions"], r["state_vars"], r["call_sites"])
    X = build_feature_matrix(
        ns["V"], ns["V_f"], ns["V_s"], ns["V_c"],
        r["functions"], r["state_vars"], r["call_sites"], r["cfg"],
    )
    return r, ns, X


class TestFeatureDimensions:
    def test_feature_dim_sum(self):
        assert FEATURE_DIM == N_FUNC_FEATURES + N_STATE_FEATURES + N_CALL_FEATURES

    def test_feature_dim_is_34(self):
        """Expected total after adding reentrancy-specific features.
        Spec: Section 4.3 (extended) — 9 + 14 + 11 = 34."""
        assert FEATURE_DIM == 34

    def test_state_dim_includes_reentrancy_flags(self):
        assert N_STATE_FEATURES == 14

    def test_call_dim_includes_reentrancy_flags(self):
        assert N_CALL_FEATURES == 11

    def test_feature_config_matches(self):
        config = get_feature_config()
        assert config["feature_dim"] == FEATURE_DIM
        assert config["state_var_features"]["size"] == N_STATE_FEATURES
        assert config["call_site_features"]["size"] == N_CALL_FEATURES
        assert "reentrancy_flags" in config["call_site_features"]
        assert "reentrancy_counts" in config["call_site_features"]


class TestWithdrawFeatures:
    """Feature matrix tests using the worked example from spec Section 4.7."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r, self.ns, self.X = _extract_and_build(WITHDRAW_SOL)

    def test_shape(self):
        assert self.X.shape == (len(self.ns["V"]), FEATURE_DIM)

    def test_no_nan(self):
        assert not np.any(np.isnan(self.X))

    def test_no_inf(self):
        assert not np.any(np.isinf(self.X))

    def test_func_features_in_func_slots(self):
        """V_f nodes should only have nonzero values in the func feature slots."""
        for node in self.ns["V_f"]:
            idx = self.ns["node_index"][node]
            row = self.X[idx]
            # Func features: indices 0..N_FUNC_FEATURES-1
            assert np.any(row[:N_FUNC_FEATURES] != 0), f"{node} has no func features"
            # State var and call site slots should be zero
            assert np.all(row[N_FUNC_FEATURES:] == 0), f"{node} has nonzero in non-func slots"

    def test_state_var_features_in_var_slots(self):
        """V_s nodes should only have nonzero values in the state var feature slots."""
        for node in self.ns["V_s"]:
            idx = self.ns["node_index"][node]
            row = self.X[idx]
            # Func slots should be zero
            assert np.all(row[:N_FUNC_FEATURES] == 0)
            # State var features: indices N_FUNC_FEATURES..N_FUNC_FEATURES+N_STATE_FEATURES-1
            sv_start = N_FUNC_FEATURES
            sv_end = N_FUNC_FEATURES + N_STATE_FEATURES
            assert np.any(row[sv_start:sv_end] != 0), f"{node} has no state var features"
            # Call site slots should be zero
            assert np.all(row[sv_end:] == 0)

    def test_call_site_features_in_call_slots(self):
        """V_c nodes should only have nonzero values in the call site feature slots."""
        for node in self.ns["V_c"]:
            idx = self.ns["node_index"][node]
            row = self.X[idx]
            # Func and state var slots should be zero
            cs_start = N_FUNC_FEATURES + N_STATE_FEATURES
            assert np.all(row[:cs_start] == 0)
            # Call site features should have something
            assert np.any(row[cs_start:] != 0), f"{node} has no call site features"

    def test_deposit_is_public_payable(self):
        idx = self.ns["node_index"]["func:deposit"]
        row = self.X[idx]
        # visibility: public = index 0
        assert row[0] == 1.0
        # mutability: payable = index 4+2=6
        assert row[6] == 1.0

    def test_withdraw_is_public_nonpayable(self):
        idx = self.ns["node_index"]["func:withdraw"]
        row = self.X[idx]
        assert row[0] == 1.0  # public
        assert row[7] == 1.0  # nonpayable = index 4+3=7

    def test_balance_is_mapping_read_write(self):
        idx = self.ns["node_index"]["var:balance"]
        row = self.X[idx]
        # type: mapping = TYPE_CATEGORIES[4], at offset 9+4=13
        assert row[13] == 1.0
        # access_pattern: read_write = ACCESS_PATTERNS[2], at offset 9+8+1+2=20
        assert row[20] == 1.0

    def test_call_site_is_call_with_value(self):
        cs_node = self.ns["V_c"][0]
        idx = self.ns["node_index"][cs_node]
        row = self.X[idx]
        cs_start = N_FUNC_FEATURES + N_STATE_FEATURES
        # opcode: call = CALL_OPCODES[0], at cs_start+0
        assert row[cs_start] == 1.0
        # value_transfer at cs_start+4
        assert row[cs_start + 4] == 1.0

    def test_call_site_reentrancy_flags(self):
        """withdraw's call site should have the reentrancy signal bits set."""
        cs_node = self.ns["V_c"][0]
        idx = self.ns["node_index"][cs_node]
        row = self.X[idx]
        cs_start = N_FUNC_FEATURES + N_STATE_FEATURES
        # Layout after cs_start: opcode(4) | value(1) | gas_forwarded(1)
        # | sender_controlled(1) | guarded(1) | writes_after(1) | reads_after(1) | reads_before(1)
        assert row[cs_start + 5] == 1.0  # gas_forwarded (.call forwards all gas)
        assert row[cs_start + 6] == 1.0  # sender_controlled_target (msg.sender.call)
        assert row[cs_start + 7] == 0.0  # guarded_by_modifier: withdraw has no guard
        # log1p(>=1) > 0 — there is at least one state var written after the call.
        assert row[cs_start + 8] > 0.0

    def test_state_var_reentrancy_flags(self):
        """balance is both written after a call and read before one."""
        idx = self.ns["node_index"]["var:balance"]
        row = self.X[idx]
        # State-var layout: offset 9 = N_FUNC_FEATURES
        # type(8) | slot(1) | access(3) | written_after_call(1) | read_before_call(1)
        written_after_idx = N_FUNC_FEATURES + 8 + 1 + 3  # 21
        read_before_idx = written_after_idx + 1  # 22
        assert row[written_after_idx] == 1.0
        assert row[read_before_idx] == 1.0


class TestStateVarReentrancyAggregation:
    def test_aggregates_across_call_sites(self):
        state_vars = [{"name": "balance"}, {"name": "other"}, {"name": "unused"}]
        call_sites = [
            {"writes_after_call": ["balance"], "reads_before_call": ["balance"]},
            {"writes_after_call": [], "reads_before_call": ["other"]},
        ]
        written, read_before = _compute_state_var_reentrancy_flags(call_sites, state_vars)
        assert written["balance"] is True
        assert written["other"] is False
        assert written["unused"] is False
        assert read_before["balance"] is True
        assert read_before["other"] is True
        assert read_before["unused"] is False


class TestTypeClassification:
    def test_uint(self):
        assert _classify_solidity_type("uint256") == "uint"
        assert _classify_solidity_type("int256") == "uint"
        assert _classify_solidity_type("uint8") == "uint"

    def test_address(self):
        assert _classify_solidity_type("address") == "address"

    def test_bool(self):
        assert _classify_solidity_type("bool") == "bool"

    def test_bytes(self):
        assert _classify_solidity_type("bytes32") == "bytes"
        assert _classify_solidity_type("string") == "bytes"

    def test_mapping(self):
        assert _classify_solidity_type("mapping(address => uint256)") == "mapping"

    def test_array(self):
        assert _classify_solidity_type("address[]") == "array"
        assert _classify_solidity_type("uint256[]") == "array"

    def test_struct(self):
        assert _classify_solidity_type("LogFile") == "struct"
        assert _classify_solidity_type("F3Ddatasets.Player") == "struct"


class TestNoCallsFeatures:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.r, self.ns, self.X = _extract_and_build(NO_CALLS_SOL)

    def test_shape(self):
        assert self.X.shape == (len(self.ns["V"]), FEATURE_DIM)

    def test_no_call_site_rows(self):
        """No V_c nodes, so no rows should have call site features."""
        cs_start = N_FUNC_FEATURES + N_STATE_FEATURES
        assert np.all(self.X[:, cs_start:] == 0)


class TestDatasetFeatures:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(REENTRANT_SOL):
            pytest.skip("Dataset not available")
        self.r, self.ns, self.X = _extract_and_build(REENTRANT_SOL, "PERSONAL_BANK")

    def test_shape(self):
        assert self.X.shape == (len(self.ns["V"]), FEATURE_DIM)

    def test_no_nan_inf(self):
        assert not np.any(np.isnan(self.X))
        assert not np.any(np.isinf(self.X))

    def test_balances_is_mapping(self):
        idx = self.ns["node_index"]["var:balances"]
        row = self.X[idx]
        # mapping type at index 13
        assert row[N_FUNC_FEATURES + 4] == 1.0

    def test_collect_call_has_value(self):
        """Collect's low-level call should have value_transfer=1."""
        cs_node = "call:Collect:47"
        idx = self.ns["node_index"][cs_node]
        row = self.X[idx]
        cs_start = N_FUNC_FEATURES + N_STATE_FEATURES
        assert row[cs_start + 4] == 1.0  # value_transfer
