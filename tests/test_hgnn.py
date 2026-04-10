"""
Tests for Step 6 — HGNN Model.
Uses the withdraw example from spec Section 4.7.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.extraction.ast_cfg import extract_all
from src.extraction.gdep import build_gdep
from src.hypergraph.features import FEATURE_DIM, build_feature_matrix
from src.hypergraph.hyperedges import build_hyperedges
from src.hypergraph.nodeset import build_node_sets
from src.model.hgnn import HGNN

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
    """Run Steps 1-5 and return all intermediate results."""
    r = extract_all(sol_path, contract_name=contract_name)
    assert r is not None
    G_dep = build_gdep(r["cfg"], r["call_sites"], r["state_vars"])
    ns = build_node_sets(r["functions"], r["state_vars"], r["call_sites"])
    X = build_feature_matrix(
        ns["V"], ns["V_f"], ns["V_s"], ns["V_c"],
        r["functions"], r["state_vars"], r["call_sites"], r["cfg"],
    )
    E, H_inc = build_hyperedges(
        ns["V"], ns["V_c"], ns["V_s"], ns["node_index"],
        r["G_call"], G_dep, r["call_sites"],
    )
    return r, G_dep, ns, X, E, H_inc


class TestHGNNBasic:
    """Basic model construction and forward pass tests."""

    def test_construction_default(self):
        model = HGNN(in_dim=26, hidden_dim=64)
        assert model.n_layers == 2
        assert model.use_layernorm is True
        assert model.hidden_dim == 64

    def test_construction_custom_layers(self):
        model = HGNN(in_dim=26, hidden_dim=32, n_layers=4)
        assert model.n_layers == 4
        assert len(model.layers) == 4

    def test_construction_no_layernorm(self):
        model = HGNN(in_dim=26, hidden_dim=32, use_layernorm=False)
        assert model.layer_norms is None

    def test_input_projection_when_dims_differ(self):
        model = HGNN(in_dim=26, hidden_dim=64)
        assert model.input_proj is not None

    def test_no_input_projection_when_dims_match(self):
        model = HGNN(in_dim=64, hidden_dim=64)
        assert model.input_proj is None

    def test_parameter_count_increases_with_layers(self):
        model_2 = HGNN(in_dim=26, hidden_dim=32, n_layers=2)
        model_4 = HGNN(in_dim=26, hidden_dim=32, n_layers=4)
        params_2 = sum(p.numel() for p in model_2.parameters())
        params_4 = sum(p.numel() for p in model_4.parameters())
        assert params_4 > params_2


class TestHGNNSyntheticForward:
    """Forward pass tests with synthetic data."""

    def test_forward_shape(self):
        """Output shape should be (|E|, 2)."""
        n_nodes, n_edges, d = 5, 2, 10
        model = HGNN(in_dim=d, hidden_dim=16)
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, n_edges)
        H_inc[0, 0] = 1.0
        H_inc[1, 0] = 1.0
        H_inc[2, 1] = 1.0
        H_inc[3, 1] = 1.0
        E = [{"n0", "n1"}, {"n2", "n3"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}
        y_pred = model(X, H_inc, E, node_index)
        assert y_pred.shape == (n_edges, 2)

    def test_output_sums_to_one(self):
        """Each row of softmax output should sum to 1."""
        n_nodes, n_edges, d = 5, 2, 10
        model = HGNN(in_dim=d, hidden_dim=16)
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, n_edges)
        H_inc[0, 0] = 1.0
        H_inc[1, 0] = 1.0
        H_inc[2, 1] = 1.0
        E = [{"n0", "n1"}, {"n2"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}
        y_pred = model(X, H_inc, E, node_index)
        row_sums = y_pred.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n_edges), atol=1e-5)

    def test_output_probabilities_valid(self):
        """All probabilities should be in [0, 1]."""
        n_nodes, n_edges, d = 4, 1, 8
        model = HGNN(in_dim=d, hidden_dim=16)
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, n_edges)
        H_inc[0, 0] = 1.0
        H_inc[1, 0] = 1.0
        E = [{"n0", "n1"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}
        y_pred = model(X, H_inc, E, node_index)
        assert (y_pred >= 0).all()
        assert (y_pred <= 1).all()

    def test_logits_shape(self):
        """forward_logits should return (|E|, 2) raw logits."""
        n_nodes, n_edges, d = 5, 2, 10
        model = HGNN(in_dim=d, hidden_dim=16)
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, n_edges)
        H_inc[0, 0] = 1.0
        H_inc[1, 0] = 1.0
        E = [{"n0", "n1"}, {"n2"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}
        logits = model.forward_logits(X, H_inc, E, node_index)
        assert logits.shape == (n_edges, 2)

    def test_gradient_flows(self):
        """Loss.backward() should produce gradients for all parameters."""
        n_nodes, n_edges, d = 5, 2, 10
        model = HGNN(in_dim=d, hidden_dim=16)
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, n_edges)
        H_inc[0, 0] = 1.0
        H_inc[1, 0] = 1.0
        H_inc[2, 1] = 1.0
        E = [{"n0", "n1"}, {"n2"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}
        labels = torch.tensor([0, 1], dtype=torch.long)

        logits = model.forward_logits(X, H_inc, E, node_index)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_single_hyperedge(self):
        """Model works with a single hyperedge."""
        n_nodes, d = 3, 8
        model = HGNN(in_dim=d, hidden_dim=16)
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, 1)
        H_inc[0, 0] = 1.0
        H_inc[1, 0] = 1.0
        E = [{"n0", "n1"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}
        y_pred = model(X, H_inc, E, node_index)
        assert y_pred.shape == (1, 2)

    def test_isolated_nodes_handled(self):
        """Nodes not in any hyperedge should not crash the model."""
        n_nodes, d = 5, 8
        model = HGNN(in_dim=d, hidden_dim=16)
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, 1)
        H_inc[0, 0] = 1.0  # Only node 0 in hyperedge
        E = [{"n0"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}
        y_pred = model(X, H_inc, E, node_index)
        assert y_pred.shape == (1, 2)

    def test_dropout_changes_output_in_train_mode(self):
        """With dropout > 0, training mode should produce non-deterministic output."""
        torch.manual_seed(42)
        n_nodes, n_edges, d = 5, 2, 10
        model = HGNN(in_dim=d, hidden_dim=16, dropout=0.5)
        model.train()
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, n_edges)
        H_inc[0, 0] = 1.0
        H_inc[1, 0] = 1.0
        H_inc[2, 1] = 1.0
        E = [{"n0", "n1"}, {"n2"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}

        torch.manual_seed(1)
        y1 = model(X, H_inc, E, node_index)
        torch.manual_seed(2)
        y2 = model(X, H_inc, E, node_index)
        # With dropout, outputs should differ across different seeds
        assert not torch.allclose(y1, y2, atol=1e-6)

    def test_eval_mode_deterministic(self):
        """In eval mode, output should be deterministic."""
        n_nodes, n_edges, d = 5, 2, 10
        model = HGNN(in_dim=d, hidden_dim=16, dropout=0.5)
        model.eval()
        X = torch.randn(n_nodes, d)
        H_inc = torch.zeros(n_nodes, n_edges)
        H_inc[0, 0] = 1.0
        H_inc[1, 0] = 1.0
        E = [{"n0", "n1"}, {"n2"}]
        node_index = {f"n{i}": i for i in range(n_nodes)}
        y1 = model(X, H_inc, E, node_index)
        y2 = model(X, H_inc, E, node_index)
        assert torch.allclose(y1, y2)


class TestHGNNWithdraw:
    """End-to-end test with the withdraw example from spec Section 4.7."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r, self.G_dep, self.ns, self.X, self.E, self.H_inc = _full_pipeline(
            WITHDRAW_SOL
        )
        self.X_t = torch.tensor(self.X, dtype=torch.float32)
        self.H_t = torch.tensor(self.H_inc, dtype=torch.float32)

    def test_forward_runs(self):
        """Forward pass completes without error on withdraw example."""
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32)
        y_pred = model(self.X_t, self.H_t, self.E, self.ns["node_index"])
        assert y_pred is not None

    def test_output_shape(self):
        """Output shape is (|E|, 2) = (1, 2) for withdraw."""
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32)
        y_pred = model(self.X_t, self.H_t, self.E, self.ns["node_index"])
        assert y_pred.shape == (1, 2)

    def test_output_sums_to_one(self):
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32)
        y_pred = model(self.X_t, self.H_t, self.E, self.ns["node_index"])
        assert torch.allclose(y_pred.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_gradient_flows_on_withdraw(self):
        """Gradient flows through full pipeline from withdraw example."""
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32)
        logits = model.forward_logits(self.X_t, self.H_t, self.E, self.ns["node_index"])
        labels = torch.tensor([1], dtype=torch.long)  # vulnerable
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_weighted_loss_works(self):
        """Weighted CrossEntropyLoss (for class imbalance) works."""
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32)
        logits = model.forward_logits(self.X_t, self.H_t, self.E, self.ns["node_index"])
        labels = torch.tensor([1], dtype=torch.long)
        weights = torch.tensor([1.0, 2.57], dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_different_layer_counts(self):
        """Model works with L=2, 3, 4."""
        for L in [2, 3, 4]:
            model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32, n_layers=L)
            y_pred = model(self.X_t, self.H_t, self.E, self.ns["node_index"])
            assert y_pred.shape == (1, 2), f"Failed with L={L}"


class TestHGNNDataset:
    """Integration test on PERSONAL_BANK from the dataset."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(REENTRANT_SOL):
            pytest.skip("Dataset not available")
        self.r, self.G_dep, self.ns, self.X, self.E, self.H_inc = _full_pipeline(
            REENTRANT_SOL, "PERSONAL_BANK"
        )
        self.X_t = torch.tensor(self.X, dtype=torch.float32)
        self.H_t = torch.tensor(self.H_inc, dtype=torch.float32)

    def test_forward_runs(self):
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=64)
        y_pred = model(self.X_t, self.H_t, self.E, self.ns["node_index"])
        assert y_pred is not None

    def test_output_shape(self):
        """PERSONAL_BANK has 3 call sites, so output should be (3, 2)."""
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=64)
        y_pred = model(self.X_t, self.H_t, self.E, self.ns["node_index"])
        assert y_pred.shape == (3, 2)

    def test_all_rows_sum_to_one(self):
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=64)
        y_pred = model(self.X_t, self.H_t, self.E, self.ns["node_index"])
        row_sums = y_pred.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)

    def test_training_step(self):
        """Full training step (forward + backward + optimizer step) works."""
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        labels = torch.tensor([1, 1, 1], dtype=torch.long)  # all vulnerable
        weights = torch.tensor([1.0, 2.57], dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        model.train()
        logits = model.forward_logits(self.X_t, self.H_t, self.E, self.ns["node_index"])
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify parameters changed
        logits_after = model.forward_logits(
            self.X_t, self.H_t, self.E, self.ns["node_index"]
        )
        assert not torch.allclose(logits, logits_after)
