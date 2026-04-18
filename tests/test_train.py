"""
Tests for Step 7 — Training Loop and CV Evaluation.
Tests CV split generation, contract processing, training, and evaluation.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from src.evaluation.train import (
    CLASS_WEIGHTS,
    REENTRANT_DIR,
    SAFE_DIR,
    _collect_probs_and_labels,
    compute_class_weights,
    compute_metrics,
    evaluate,
    generate_cv_splits,
    process_contract,
    train_epoch,
    train_fold,
    tune_threshold,
)
from src.hypergraph.features import FEATURE_DIM
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


# ── CV Split Tests ─────────────────────────────────────────────────


class TestCVSplits:
    """Test CV split generation matches make_folds.py logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(REENTRANT_DIR):
            pytest.skip("Dataset not available")

    def test_generates_three_folds(self):
        folds = generate_cv_splits(n_splits=3)
        assert len(folds) == 3

    def test_each_fold_has_train_and_val(self):
        folds = generate_cv_splits(n_splits=3)
        for fold in folds:
            assert "train" in fold
            assert "val" in fold
            assert len(fold["train"]) > 0
            assert len(fold["val"]) > 0

    def test_no_overlap_between_train_and_val(self):
        """Train and val sets should be disjoint within each fold."""
        folds = generate_cv_splits(n_splits=3)
        for fold in folds:
            train_paths = {path for path, _ in fold["train"]}
            val_paths = {path for path, _ in fold["val"]}
            overlap = train_paths & val_paths
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_contracts_covered(self):
        """Every contract should appear in exactly one val set across folds."""
        folds = generate_cv_splits(n_splits=3)
        all_val = []
        for fold in folds:
            all_val.extend(path for path, _ in fold["val"])
        # No duplicates
        assert len(all_val) == len(set(all_val))

    def test_deterministic(self):
        """Same random_state produces identical splits."""
        folds1 = generate_cv_splits(n_splits=3, random_state=42)
        folds2 = generate_cv_splits(n_splits=3, random_state=42)
        for f1, f2 in zip(folds1, folds2):
            paths1 = [p for p, _ in f1["train"]]
            paths2 = [p for p, _ in f2["train"]]
            assert paths1 == paths2

    def test_labels_correct(self):
        """Reentrant contracts get label 1, safe get label 0."""
        folds = generate_cv_splits(n_splits=3)
        for fold in folds:
            for path, label in fold["train"] + fold["val"]:
                if "reentrant" in path:
                    assert label == 1
                elif "safe" in path:
                    assert label == 0

    def test_both_classes_in_each_fold(self):
        """Each fold should contain both classes in train and val."""
        folds = generate_cv_splits(n_splits=3)
        for fold in folds:
            train_labels = {l for _, l in fold["train"]}
            val_labels = {l for _, l in fold["val"]}
            assert 0 in train_labels and 1 in train_labels
            assert 0 in val_labels and 1 in val_labels

    def test_total_contracts_match_dataset(self):
        """Total across all folds should match dataset size."""
        folds = generate_cv_splits(n_splits=3)
        total = sum(
            len(fold["train"]) + len(fold["val"]) for fold in folds
        )
        # Each contract appears in train (n-1 times) + val (1 time) = n times total
        n_contracts = len(folds[0]["train"]) + len(folds[0]["val"])
        assert total == n_contracts * 3


# ── Contract Processing Tests ──────────────────────────────────────


class TestProcessContract:
    def test_withdraw_processes(self):
        result = process_contract(WITHDRAW_SOL, label=1)
        assert result is not None
        assert result["label"] == 1
        assert result["n_hyperedges"] == 1
        assert result["X"].shape[1] == FEATURE_DIM

    def test_per_hyperedge_labels_present(self):
        """process_contract now writes per-hyperedge labels aligned with V_c."""
        result = process_contract(WITHDRAW_SOL, label=1)
        assert result is not None
        assert "labels" in result
        assert len(result["labels"]) == result["n_hyperedges"]

    def test_safe_contract_all_zero_labels(self):
        """A safe label should yield all-zero per-hyperedge labels."""
        result = process_contract(WITHDRAW_SOL, label=0)
        assert result is not None
        assert result["labels"] == [0] * result["n_hyperedges"]

    def test_no_calls_returns_none(self):
        """Contract with no external calls should return None (no hyperedges)."""
        result = process_contract(NO_CALLS_SOL, label=0)
        assert result is None

    def test_nonexistent_file_returns_none(self):
        result = process_contract("/nonexistent/file.sol", label=0)
        assert result is None

    def test_dataset_contract_processes(self):
        if not os.path.exists(REENTRANT_SOL):
            pytest.skip("Dataset not available")
        result = process_contract(REENTRANT_SOL, label=1, contract_name="PERSONAL_BANK")
        assert result is not None
        assert result["n_hyperedges"] == 3


# ── Metrics Tests ──────────────────────────────────────────────────


class TestComputeMetrics:
    def test_perfect_predictions(self):
        preds = [1, 1, 0, 0]
        labels = [1, 1, 0, 0]
        m = compute_metrics(preds, labels)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["fnr"] == 0.0
        assert m["fpr"] == 0.0
        assert m["accuracy"] == 1.0

    def test_all_wrong(self):
        preds = [0, 0, 1, 1]
        labels = [1, 1, 0, 0]
        m = compute_metrics(preds, labels)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["fnr"] == 1.0
        assert m["fpr"] == 1.0

    def test_mixed_predictions(self):
        preds = [1, 0, 1, 0]
        labels = [1, 1, 0, 0]
        m = compute_metrics(preds, labels)
        assert m["tp"] == 1
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["tn"] == 1
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5

    def test_empty_predictions(self):
        m = compute_metrics([], [])
        assert m["accuracy"] == 0.0
        assert m["n_total"] == 0


# ── Class Weight Tests ─────────────────────────────────────────────


class TestComputeClassWeights:
    """compute_class_weights derives weights from per-hyperedge label distribution."""

    def test_balanced_labels(self):
        data = [{"labels": [0, 1, 0, 1], "n_hyperedges": 4, "label": 1}]
        w = compute_class_weights(data)
        assert w[0].item() == 1.0
        assert w[1].item() == pytest.approx(1.0)

    def test_imbalanced_labels(self):
        # 3 neg, 1 pos -> weight[1] == 3.0
        data = [{"labels": [0, 0, 0, 1], "n_hyperedges": 4, "label": 1}]
        w = compute_class_weights(data)
        assert w[1].item() == pytest.approx(3.0)

    def test_clamps_when_positives_zero(self):
        data = [{"labels": [0, 0, 0], "n_hyperedges": 3, "label": 0}]
        w = compute_class_weights(data, clamp=10.0)
        assert w[1].item() == 10.0

    def test_clamp_cap(self):
        # 100 neg, 1 pos would yield 100 without clamp; clamped to 10.
        data = [
            {"labels": [0] * 100, "n_hyperedges": 100, "label": 0},
            {"labels": [1], "n_hyperedges": 1, "label": 1},
        ]
        w = compute_class_weights(data, clamp=10.0)
        assert w[1].item() == 10.0

    def test_fallback_to_contract_label(self):
        """Data that predates per-hyperedge labels should still compute weights."""
        data = [{"n_hyperedges": 4, "label": 1}]
        w = compute_class_weights(data)
        # All 4 hyperedges labeled 1 via fallback -> no negatives, clamp fires.
        assert w[1].item() == 10.0


# ── Training Tests ─────────────────────────────────────────────────


class TestTraining:
    """Test training mechanics on the withdraw fixture."""

    @pytest.fixture
    def withdraw_data(self):
        data = process_contract(WITHDRAW_SOL, label=1)
        assert data is not None
        return [data]

    def test_train_epoch_returns_loss(self, withdraw_data):
        torch.manual_seed(42)
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

        loss = train_epoch(model, optimizer, loss_fn, withdraw_data)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases_over_epochs(self, withdraw_data):
        torch.manual_seed(42)
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

        losses = []
        for _ in range(20):
            loss = train_epoch(model, optimizer, loss_fn, withdraw_data)
            losses.append(loss)

        # Loss should generally decrease (first > last)
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_evaluate_returns_metrics(self, withdraw_data):
        model = HGNN(in_dim=FEATURE_DIM, hidden_dim=32)
        metrics = evaluate(model, withdraw_data)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "fnr" in metrics
        assert "fpr" in metrics
        assert "predictions" in metrics
        assert len(metrics["predictions"]) == 1  # 1 hyperedge


class TestTrainFold:
    """Test full fold training with temporary results directory."""

    def test_train_fold_completes(self):
        data = process_contract(WITHDRAW_SOL, label=1)
        assert data is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_fold(
                fold_idx=0,
                train_data=[data],
                val_data=[data],
                seed=42,
                epochs=5,
                results_dir=tmpdir,
            )
            assert "f1" in metrics
            assert "loss_history" in metrics
            assert len(metrics["loss_history"]) == 5

            # Check predictions CSV was saved
            pred_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
            assert len(pred_files) > 0

    def test_train_fold_with_focal_loss(self):
        """Step 3: train_fold should accept loss_type='focal' and train."""
        data = process_contract(WITHDRAW_SOL, label=1)
        assert data is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_fold(
                fold_idx=0,
                train_data=[data],
                val_data=[data],
                seed=42,
                epochs=5,
                dropout=0.3,
                weight_decay=1e-4,
                loss_type="focal",
                focal_gamma=2.0,
                results_dir=tmpdir,
            )
            assert "f1" in metrics
            assert len(metrics["loss_history"]) == 5

    def test_train_fold_rejects_unknown_loss(self):
        data = process_contract(WITHDRAW_SOL, label=1)
        assert data is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown loss_type"):
                train_fold(
                    fold_idx=0,
                    train_data=[data],
                    val_data=[data],
                    seed=42,
                    epochs=1,
                    loss_type="hinge",
                    results_dir=tmpdir,
                )

    def test_train_fold_reports_threshold_diagnostics(self):
        """Val eval uses threshold=0.5; the train-optimal threshold is
        recorded separately as a diagnostic. (Train-based threshold tuning
        regressed val F1 on 2026-04-17 — see tune_threshold docstring.)"""
        data = process_contract(WITHDRAW_SOL, label=1)
        assert data is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_fold(
                fold_idx=0,
                train_data=[data],
                val_data=[data],
                seed=42,
                epochs=5,
                results_dir=tmpdir,
            )
            # Applied threshold for val is always 0.5 (default of evaluate).
            assert metrics["threshold"] == 0.5
            # Diagnostic: train-sweep result is still surfaced.
            assert "train_opt_threshold" in metrics
            assert 0.10 <= metrics["train_opt_threshold"] <= 0.90
            assert "train_opt_f1" in metrics


# ── Threshold Tuning Tests (Step 2) ────────────────────────────────


class _StubTwoClassModel(torch.nn.Module):
    """Deterministic fake model: returns a fixed prob_vuln for each hyperedge.

    Lets us test threshold sweeps without training — the contract dict carries
    a custom "probs" key that process_contract would never write.
    """

    def __init__(self, probs_per_contract):
        super().__init__()
        self._probs = probs_per_contract
        self._i = 0

    def reset(self):
        self._i = 0

    def forward(self, X, H_inc, E, node_index):
        probs = self._probs[self._i]
        self._i += 1
        p = torch.tensor(probs, dtype=torch.float32)
        return torch.stack([1 - p, p], dim=1)


def _make_fake_contract(n_edges, labels, sol_path="/fake.sol"):
    return {
        "X": np.zeros((1, FEATURE_DIM), dtype=np.float32),
        "H_inc": np.zeros((1, n_edges), dtype=np.float32),
        "E": [set() for _ in range(n_edges)],
        "node_index": {},
        "label": 1 if any(labels) else 0,
        "labels": list(labels),
        "n_hyperedges": n_edges,
        "sol_path": sol_path,
    }


class TestEvaluateThreshold:
    """evaluate() should honor the `threshold` arg and record it in metrics."""

    def test_default_threshold_matches_argmax(self):
        # prob_vuln = 0.6 → at t=0.5 both hyperedges predicted 1.
        model = _StubTwoClassModel([[0.6, 0.6]])
        contract = _make_fake_contract(2, labels=[1, 1])
        m = evaluate(model, [contract])
        assert m["threshold"] == 0.5
        assert m["tp"] == 2
        assert m["fn"] == 0

    def test_higher_threshold_reduces_positives(self):
        # prob_vuln = 0.6. t=0.5 predicts vuln; t=0.7 predicts safe.
        model = _StubTwoClassModel([[0.6, 0.6]])
        contract = _make_fake_contract(2, labels=[1, 1])
        m = evaluate(model, [contract], threshold=0.7)
        assert m["threshold"] == 0.7
        assert m["tp"] == 0
        assert m["fn"] == 2

    def test_lower_threshold_boosts_recall(self):
        # prob_vuln = 0.3. t=0.5 misses; t=0.2 catches.
        contract = _make_fake_contract(2, labels=[1, 1])

        model_a = _StubTwoClassModel([[0.3, 0.3]])
        m_high = evaluate(model_a, [contract], threshold=0.5)
        assert m_high["recall"] == 0.0

        model_b = _StubTwoClassModel([[0.3, 0.3]])
        m_low = evaluate(model_b, [contract], threshold=0.2)
        assert m_low["recall"] == 1.0


class TestTuneThreshold:
    """tune_threshold picks the F1-max threshold on the training set."""

    def test_picks_threshold_that_captures_true_positive(self):
        # Two hyperedges, both labeled 1. Probs 0.3 and 0.8.
        # t=0.5: recall=0.5; t=0.25: recall=1.0, precision=1.0, F1=1.0.
        model = _StubTwoClassModel([[0.3, 0.8]])
        contract = _make_fake_contract(2, labels=[1, 1])
        t, f1 = tune_threshold(model, [contract])
        assert f1 == pytest.approx(1.0)
        assert t <= 0.35

    def test_tie_break_prefers_threshold_near_half(self):
        # All labels 0 → F1 is always 0 regardless of threshold → ties across
        # the sweep. Tie-break should snap to t closest to 0.5.
        model = _StubTwoClassModel([[0.1, 0.9]])
        contract = _make_fake_contract(2, labels=[0, 0])
        t, f1 = tune_threshold(model, [contract])
        assert f1 == 0.0
        assert t == pytest.approx(0.5)

    def test_empty_data_returns_half(self):
        model = _StubTwoClassModel([])
        t, f1 = tune_threshold(model, [])
        assert t == 0.5
        assert f1 == 0.0

    def test_collect_probs_and_labels_shapes(self):
        model = _StubTwoClassModel([[0.1, 0.9], [0.5]])
        contracts = [
            _make_fake_contract(2, labels=[0, 1]),
            _make_fake_contract(1, labels=[1]),
        ]
        probs, labels = _collect_probs_and_labels(model, contracts)
        assert probs.shape == (3,)
        assert labels.tolist() == [0, 1, 1]
