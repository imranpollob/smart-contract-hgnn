"""
Step 7 — Training Loop and CV Evaluation.
Spec: Section 7 (Evaluation Plan).

Implements 3-fold cross-validation on the aggregated benchmark,
with weighted CrossEntropyLoss, multi-seed evaluation, and metric logging.
"""

import csv
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from src.extraction.ast_cfg import extract_all
from src.extraction.gdep import build_gdep
from src.hypergraph.features import FEATURE_DIM, build_feature_matrix
from src.hypergraph.hyperedges import build_hyperedges
from src.hypergraph.nodeset import build_node_sets
from src.model.hgnn import HGNN

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "reentrancy-detection-benchmarks",
    "benchmarks",
    "aggregated-benchmark",
    "src",
)
REENTRANT_DIR = os.path.join(DATA_ROOT, "reentrant")
SAFE_DIR = os.path.join(DATA_ROOT, "safe")

DEFAULT_SEEDS = [42, 0, 1, 2, 3]
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN_DIM = 64
DEFAULT_N_LAYERS = 2

# Class weights: safe=1.0, reentrant=2.57 (314/122 ≈ 2.57)
CLASS_WEIGHTS = torch.tensor([1.0, 2.57], dtype=torch.float32)


# ── CV Split Generation ───────────────────────────────────────────


def generate_cv_splits(n_splits: int = 3, random_state: int = 42) -> list[dict]:
    """
    Generate stratified CV splits matching make_folds.py logic.
    Spec: Section 7 — KFold(n_splits=3, shuffle=True, random_state=42) per class.

    Returns:
        List of dicts, one per fold: {"train": [...], "val": [...]}
        Each entry is (sol_path, label) where label=1 for reentrant, 0 for safe.
    """
    # Collect and sort files per class (matching make_folds.py)
    reentrant_files = sorted([
        f for f in os.listdir(REENTRANT_DIR)
        if f.endswith(".sol") and os.path.isfile(os.path.join(REENTRANT_DIR, f))
    ])
    safe_files = sorted([
        f for f in os.listdir(SAFE_DIR)
        if f.endswith(".sol") and os.path.isfile(os.path.join(SAFE_DIR, f))
    ])

    logger.info(f"Dataset: {len(reentrant_files)} reentrant, {len(safe_files)} safe")

    # Apply KFold per class separately (matches make_folds.py)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    reentrant_splits = list(kf.split(reentrant_files))
    safe_splits = list(kf.split(safe_files))

    folds = []
    for fold_idx in range(n_splits):
        train_re_idx, val_re_idx = reentrant_splits[fold_idx]
        train_safe_idx, val_safe_idx = safe_splits[fold_idx]

        train = []
        val = []

        for i in train_re_idx:
            train.append((os.path.join(REENTRANT_DIR, reentrant_files[i]), 1))
        for i in val_re_idx:
            val.append((os.path.join(REENTRANT_DIR, reentrant_files[i]), 1))
        for i in train_safe_idx:
            train.append((os.path.join(SAFE_DIR, safe_files[i]), 0))
        for i in val_safe_idx:
            val.append((os.path.join(SAFE_DIR, safe_files[i]), 0))

        folds.append({"train": train, "val": val})

    return folds


# ── Pipeline Processing ───────────────────────────────────────────


def process_contract(sol_path: str, label: int, contract_name: str | None = None) -> dict | None:
    """
    Run Steps 1-5 on a single contract and return all data needed for training.

    Returns:
        dict with keys: X, H_inc, E, node_index, label, n_hyperedges, sol_path
        or None if extraction fails.
    """
    try:
        r = extract_all(sol_path, contract_name=contract_name)
        if r is None:
            return None

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

        if len(E) == 0:
            # No external calls — no hyperedges to classify
            return None

        return {
            "X": X,
            "H_inc": H_inc,
            "E": E,
            "node_index": ns["node_index"],
            "label": label,
            "n_hyperedges": len(E),
            "sol_path": sol_path,
        }
    except Exception as e:
        logger.warning(f"Failed to process {sol_path}: {e}")
        return None


def process_contract_list(contracts: list[tuple[str, int]]) -> list[dict]:
    """
    Process a list of (sol_path, label) pairs. Skips failures.

    Returns:
        List of processed contract dicts (from process_contract).
    """
    results = []
    for sol_path, label in contracts:
        data = process_contract(sol_path, label)
        if data is not None:
            results.append(data)
    return results


# ── Training and Evaluation ───────────────────────────────────────


def train_epoch(
    model: HGNN,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    train_data: list[dict],
) -> float:
    """
    Train one epoch over all contracts.
    Spec: Section 7 — per-contract forward pass + loss aggregation.

    Returns:
        Average loss over all hyperedges in the epoch.
    """
    model.train()
    total_loss = 0.0
    total_edges = 0

    for contract in train_data:
        X = torch.tensor(contract["X"], dtype=torch.float32)
        H_inc = torch.tensor(contract["H_inc"], dtype=torch.float32)
        E = contract["E"]
        node_index = contract["node_index"]
        label = contract["label"]
        n_edges = contract["n_hyperedges"]

        # All hyperedges in a contract share the same label
        # Spec: Section 6.2 — contract-level label assignment
        labels = torch.full((n_edges,), label, dtype=torch.long)

        logits = model.forward_logits(X, H_inc, E, node_index)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * n_edges
        total_edges += n_edges

    return total_loss / max(total_edges, 1)


def evaluate(
    model: HGNN,
    val_data: list[dict],
) -> dict:
    """
    Evaluate model on validation data.
    Spec: Section 7 — Precision, Recall, F1, FNR, FPR.

    Returns:
        dict with metrics: precision, recall, f1, fnr, fpr, accuracy,
        and per-hyperedge predictions list.
    """
    model.eval()
    all_preds = []
    all_labels = []
    predictions = []  # For per-hyperedge CSV output

    with torch.no_grad():
        for contract in val_data:
            X = torch.tensor(contract["X"], dtype=torch.float32)
            H_inc = torch.tensor(contract["H_inc"], dtype=torch.float32)
            E = contract["E"]
            node_index = contract["node_index"]
            label = contract["label"]
            n_edges = contract["n_hyperedges"]

            y_pred = model(X, H_inc, E, node_index)
            pred_labels = y_pred.argmax(dim=1).tolist()
            true_labels = [label] * n_edges

            all_preds.extend(pred_labels)
            all_labels.extend(true_labels)

            for j, (pred, prob) in enumerate(zip(pred_labels, y_pred.tolist())):
                predictions.append({
                    "sol_path": contract["sol_path"],
                    "hyperedge_idx": j,
                    "true_label": label,
                    "pred_label": pred,
                    "prob_safe": prob[0],
                    "prob_vuln": prob[1],
                })

    return compute_metrics(all_preds, all_labels, predictions)


def compute_metrics(
    preds: list[int],
    labels: list[int],
    predictions: list[dict] | None = None,
) -> dict:
    """
    Compute classification metrics.
    Spec: Section 7 — Precision, Recall, F1, FNR, FPR.

    Args:
        preds: predicted labels (0 or 1)
        labels: true labels (0 or 1)
        predictions: optional per-hyperedge prediction dicts

    Returns:
        dict with metrics
    """
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(preds) if len(preds) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fnr": fnr,
        "fpr": fpr,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_total": len(preds),
        "predictions": predictions or [],
    }


# ── Full Training Run ─────────────────────────────────────────────


def train_fold(
    fold_idx: int,
    train_data: list[dict],
    val_data: list[dict],
    seed: int = 42,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_layers: int = DEFAULT_N_LAYERS,
    results_dir: str = "results",
) -> dict:
    """
    Train and evaluate one fold.
    Spec: Section 7 — per-fold training loop.

    Args:
        fold_idx: fold number (0-indexed)
        train_data: processed training contracts
        val_data: processed validation contracts
        seed: random seed
        epochs: number of training epochs
        lr: learning rate
        hidden_dim: HGNN hidden dimension
        n_layers: number of HGNN layers
        results_dir: directory to save results

    Returns:
        dict with final validation metrics
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HGNN(
        in_dim=FEATURE_DIM,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_layernorm=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    logger.info(
        f"Fold {fold_idx + 1} | seed={seed} | "
        f"train={len(train_data)} contracts | val={len(val_data)} contracts"
    )

    best_f1 = 0.0
    best_metrics = None
    loss_history = []

    for epoch in range(epochs):
        avg_loss = train_epoch(model, optimizer, loss_fn, train_data)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            metrics = evaluate(model, val_data)
            logger.info(
                f"  Epoch {epoch + 1:3d} | loss={avg_loss:.4f} | "
                f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                f"F1={metrics['f1']:.3f} FNR={metrics['fnr']:.3f} FPR={metrics['fpr']:.3f}"
            )
            if metrics["f1"] >= best_f1:
                best_f1 = metrics["f1"]
                best_metrics = metrics

                # Save best checkpoint
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = os.path.join(
                    "checkpoints", f"hgnn_fold{fold_idx + 1}_seed{seed}.pt"
                )
                torch.save(model.state_dict(), ckpt_path)

    # Final evaluation
    final_metrics = evaluate(model, val_data)

    # Save per-hyperedge predictions
    os.makedirs(results_dir, exist_ok=True)
    pred_path = os.path.join(
        results_dir, f"fold{fold_idx + 1}_seed{seed}_predictions.csv"
    )
    _save_predictions_csv(final_metrics["predictions"], pred_path)

    final_metrics["loss_history"] = loss_history
    return final_metrics


def run_cv(
    seeds: list[int] | None = None,
    n_splits: int = 3,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_layers: int = DEFAULT_N_LAYERS,
    results_dir: str = "results",
) -> dict:
    """
    Run full cross-validation with multiple seeds.
    Spec: Section 7 — 5 seeds [42, 0, 1, 2, 3], report mean ± std.

    Args:
        seeds: list of random seeds
        n_splits: number of CV folds
        epochs: training epochs per fold
        lr: learning rate
        hidden_dim: HGNN hidden dimension
        n_layers: number of HGNN layers
        results_dir: directory to save results

    Returns:
        dict with per-seed results and aggregated statistics
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    os.makedirs(results_dir, exist_ok=True)

    # Generate CV splits (deterministic)
    folds = generate_cv_splits(n_splits=n_splits)

    # Pre-process all contracts (shared across seeds)
    logger.info("Processing training and validation contracts...")
    fold_data = []
    for fold_idx, fold in enumerate(folds):
        logger.info(f"Processing fold {fold_idx + 1} contracts...")
        train_data = process_contract_list(fold["train"])
        val_data = process_contract_list(fold["val"])
        logger.info(
            f"  Fold {fold_idx + 1}: {len(train_data)}/{len(fold['train'])} train, "
            f"{len(val_data)}/{len(fold['val'])} val processed successfully"
        )
        fold_data.append({"train": train_data, "val": val_data})

    # Run training for each seed
    all_results = []
    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed {seed}")
        logger.info(f"{'='*60}")

        seed_metrics = []
        for fold_idx in range(n_splits):
            metrics = train_fold(
                fold_idx=fold_idx,
                train_data=fold_data[fold_idx]["train"],
                val_data=fold_data[fold_idx]["val"],
                seed=seed,
                epochs=epochs,
                lr=lr,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                results_dir=results_dir,
            )
            seed_metrics.append(metrics)

        all_results.append({"seed": seed, "fold_metrics": seed_metrics})

    # Aggregate results
    summary = _aggregate_results(all_results, results_dir)
    return summary


# ── Helpers ────────────────────────────────────────────────────────


def _save_predictions_csv(predictions: list[dict], path: str) -> None:
    """Save per-hyperedge predictions to CSV."""
    if not predictions:
        return
    keys = ["sol_path", "hyperedge_idx", "true_label", "pred_label", "prob_safe", "prob_vuln"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(predictions)


def _aggregate_results(all_results: list[dict], results_dir: str) -> dict:
    """
    Aggregate metrics across seeds and folds. Report mean ± std.
    Spec: Section 7 — 5 seeds, report mean ± std.
    """
    metric_names = ["precision", "recall", "f1", "fnr", "fpr", "accuracy"]
    all_values = {m: [] for m in metric_names}

    for result in all_results:
        for fold_metrics in result["fold_metrics"]:
            for m in metric_names:
                all_values[m].append(fold_metrics[m])

    summary = {}
    for m in metric_names:
        vals = np.array(all_values[m])
        summary[f"{m}_mean"] = float(vals.mean())
        summary[f"{m}_std"] = float(vals.std())

    # Save summary CSV
    summary_path = os.path.join(results_dir, "cv_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for m in metric_names:
            writer.writerow([m, f"{summary[f'{m}_mean']:.4f}", f"{summary[f'{m}_std']:.4f}"])

    # Save per-fold-per-seed metrics
    detail_path = os.path.join(results_dir, "cv_detailed.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "fold", *metric_names])
        for result in all_results:
            seed = result["seed"]
            for fold_idx, fold_metrics in enumerate(result["fold_metrics"]):
                row = [seed, fold_idx + 1]
                row.extend(f"{fold_metrics[m]:.4f}" for m in metric_names)
                writer.writerow(row)

    logger.info(f"\nCV Summary (mean ± std across {len(all_results)} seeds x folds):")
    for m in metric_names:
        logger.info(f"  {m:12s}: {summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}")

    summary["all_results"] = all_results
    return summary


# ── CLI Entry Point ───────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="HGNN Training + CV Evaluation")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    summary = run_cv(
        seeds=args.seeds,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        results_dir=args.results_dir,
    )
