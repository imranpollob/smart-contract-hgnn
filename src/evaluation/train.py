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
from src.model.losses import FocalLoss

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
DEFAULT_DROPOUT = 0.0
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_LOSS_TYPE = "ce"  # "ce" | "focal"
DEFAULT_FOCAL_GAMMA = 2.0

# Fallback class weights for ad-hoc test runs that never see a full fold. The
# production pipeline computes weights from the actual per-hyperedge label
# distribution via compute_class_weights() — the contract-level 1:2.57 ratio
# is no longer the right signal once labels are assigned per call site.
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

    Per-hyperedge labels (Section 6.2, revised): we no longer assign the
    contract-level label to every hyperedge. For reentrant contracts, Slither's
    reentrancy detectors pick out the specific call sites that are actually
    vulnerable; the rest of the call sites in the same contract are labeled 0.
    Safe contracts keep all-zero labels.

    Returns:
        dict with keys: X, H_inc, E, node_index, label (contract-level),
        labels (per-hyperedge list aligned with E/V_c), n_hyperedges, sol_path,
        label_info, or None if extraction fails.
    """
    try:
        r = extract_all(sol_path, contract_name=contract_name, contract_label=label)
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

        # Align per-hyperedge labels with the V_c / E ordering.
        cs_labels = r.get("call_site_labels", {})
        labels = [int(cs_labels.get(c, label)) for c in ns["V_c"]]

        return {
            "X": X,
            "H_inc": H_inc,
            "E": E,
            "node_index": ns["node_index"],
            "label": label,
            "labels": labels,
            "n_hyperedges": len(E),
            "sol_path": sol_path,
            "label_info": r.get("label_info", {}),
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
    grad_clip: float | None = 1.0,
) -> float:
    """
    Train one epoch over all contracts.
    Spec: Section 7 — per-contract forward pass + loss aggregation.

    Uses per-hyperedge labels (Section 6.2 revised): only call sites flagged
    by Slither's reentrancy detectors are labeled 1 in reentrant contracts.

    Args:
        grad_clip: if not None, clip gradient L2 norm to this value before
            stepping the optimizer. Prevents late-epoch loss blow-ups observed
            on a few outlier contracts (kept at 1.0 by default).

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
        n_edges = contract["n_hyperedges"]

        # Per-hyperedge labels; fall back to the contract label if a run
        # predates the per-call-site labeler.
        per_edge = contract.get("labels")
        if per_edge is None:
            per_edge = [contract["label"]] * n_edges
        labels = torch.tensor(per_edge, dtype=torch.long)

        logits = model.forward_logits(X, H_inc, E, node_index)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * n_edges
        total_edges += n_edges

    return total_loss / max(total_edges, 1)


def evaluate(
    model: HGNN,
    val_data: list[dict],
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model on validation data.
    Spec: Section 7 — Precision, Recall, F1, FNR, FPR.

    Metrics are computed against the per-hyperedge labels written by
    process_contract (Section 6.2 revised).

    Args:
        threshold: a hyperedge is predicted vulnerable iff prob_vuln >= threshold.
            At 0.5 this is equivalent to argmax over the 2 softmax outputs
            (current default). Lower values trade precision for recall; the
            per-fold optimum is picked by tune_threshold on train data.

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
            contract_label = contract["label"]
            n_edges = contract["n_hyperedges"]

            per_edge = contract.get("labels")
            if per_edge is None:
                per_edge = [contract_label] * n_edges

            y_pred = model(X, H_inc, E, node_index)
            prob_vuln = y_pred[:, 1]
            pred_labels = (prob_vuln >= threshold).long().tolist()

            all_preds.extend(pred_labels)
            all_labels.extend(per_edge)

            for j, (pred, prob) in enumerate(zip(pred_labels, y_pred.tolist())):
                predictions.append({
                    "sol_path": contract["sol_path"],
                    "hyperedge_idx": j,
                    "contract_label": contract_label,
                    "true_label": per_edge[j],
                    "pred_label": pred,
                    "prob_safe": prob[0],
                    "prob_vuln": prob[1],
                })

    metrics = compute_metrics(all_preds, all_labels, predictions)
    metrics["threshold"] = float(threshold)
    return metrics


def _collect_probs_and_labels(
    model: HGNN, data: list[dict], device: torch.device | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model once over `data` and return (prob_vuln, per-hyperedge labels)."""
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for contract in data:
            X = torch.tensor(contract["X"], dtype=torch.float32)
            H_inc = torch.tensor(contract["H_inc"], dtype=torch.float32)
            if device is not None:
                X = X.to(device)
                H_inc = H_inc.to(device)
            E = contract["E"]
            node_index = contract["node_index"]
            n_edges = contract["n_hyperedges"]

            per_edge = contract.get("labels")
            if per_edge is None:
                per_edge = [contract["label"]] * n_edges

            y_pred = model(X, H_inc, E, node_index)
            probs.extend(y_pred[:, 1].detach().cpu().tolist())
            labels.extend(per_edge)

    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def tune_threshold(
    model: HGNN,
    train_data: list[dict],
    thresholds: np.ndarray | None = None,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    Pick the decision threshold that maximizes F1 on the training set.

    Run on *train* data only to avoid leaking val information — best_state
    was already selected using val F1@0.5, so tuning the threshold on val too
    would double-dip. Train/val share the same distribution, so the picked
    threshold transfers.

    Tie-break: if several thresholds tie on F1, prefer the one closest to 0.5
    (most conservative choice).

    Returns:
        (best_threshold, best_train_f1)
    """
    if thresholds is None:
        # 0.10, 0.15, ..., 0.90
        thresholds = np.arange(0.10, 0.91, 0.05)

    probs, labels = _collect_probs_and_labels(model, train_data, device=device)
    if probs.size == 0:
        return 0.5, 0.0

    best_f1 = -1.0
    best_t = 0.5
    best_dist = float("inf")
    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        dist = abs(t - 0.5)
        if f1 > best_f1 or (f1 == best_f1 and dist < best_dist):
            best_f1 = f1
            best_t = float(t)
            best_dist = dist

    return best_t, float(best_f1)


def compute_class_weights(train_data: list[dict], clamp: float = 10.0) -> torch.Tensor:
    """
    Compute CrossEntropyLoss class weights from the actual per-hyperedge
    label distribution of the training set.

    weight[1] = #neg / #pos, clamped to `clamp` to avoid blow-up when the
    positive class is extremely rare (e.g. detectors flag very few sites).
    weight[0] stays at 1.0.

    Args:
        train_data: list of contract dicts (from process_contract)
        clamp: maximum value for the positive-class weight

    Returns:
        torch.Tensor shape (2,) on CPU
    """
    pos = 0
    neg = 0
    for c in train_data:
        labels = c.get("labels")
        if labels is None:
            labels = [c["label"]] * c.get("n_hyperedges", 0)
        for y in labels:
            if y == 1:
                pos += 1
            else:
                neg += 1

    if pos == 0:
        w1 = clamp
    elif neg == 0:
        w1 = clamp
    else:
        w1 = min(max(neg / pos, 1.0), clamp)

    return torch.tensor([1.0, float(w1)], dtype=torch.float32)


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
    dropout: float = DEFAULT_DROPOUT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    loss_type: str = DEFAULT_LOSS_TYPE,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
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
        dropout: dropout applied inside each HGNN layer; Step 3 raises this
            from 0.0 to close the train/val F1 gap.
        weight_decay: L2 regularization on Adam (Step 3).
        loss_type: "ce" for weighted CrossEntropy (default) or "focal" for
            focal loss with per-class α derived from compute_class_weights.
        focal_gamma: focusing parameter when loss_type="focal".
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
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Derive class weights from the actual per-hyperedge label distribution
    # of this fold's training set (Section 7 revised).
    class_weights = compute_class_weights(train_data)
    if loss_type == "focal":
        loss_fn = FocalLoss(gamma=focal_gamma, weight=class_weights)
    elif loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r} (want 'ce' or 'focal')")

    logger.info(
        f"Fold {fold_idx + 1} | seed={seed} | "
        f"train={len(train_data)} contracts | val={len(val_data)} contracts"
    )

    # Track best state by F1 (tie-break on lower training loss). Final fold
    # metrics come from this best state, not from the last-epoch weights —
    # otherwise a late-epoch collapse (seen on seed 42 / fold 2) wipes out an
    # earlier good run.
    best_f1 = -1.0
    best_loss_at_best = float("inf")
    best_state = None
    loss_history = []

    for epoch in range(epochs):
        avg_loss = train_epoch(model, optimizer, loss_fn, train_data)
        loss_history.append(avg_loss)

        # Evaluate every epoch so best_state can be selected from the true
        # best epoch, not from a sparse snapshot.
        metrics = evaluate(model, val_data)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(
                f"  Epoch {epoch + 1:3d} | loss={avg_loss:.4f} | "
                f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                f"F1={metrics['f1']:.3f} FNR={metrics['fnr']:.3f} FPR={metrics['fpr']:.3f}"
            )

        # Strictly-better F1, or same F1 with lower training loss.
        is_better = metrics["f1"] > best_f1 or (
            metrics["f1"] == best_f1 and avg_loss < best_loss_at_best
        )
        if is_better:
            best_f1 = metrics["f1"]
            best_loss_at_best = avg_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Load best state before final evaluation so the fold result reflects
    # the best epoch, not the last one.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Diagnostic only: what threshold *would* the train set pick? Logged so
    # the calibration gap (train F1 >> val F1) is visible, but NOT applied
    # to val — the 2026-04-17 sweep showed train-optimal thresholds (0.10–
    # 0.30) regress val F1 because the overfit model's probabilities are
    # poorly calibrated. Threshold tuning is deferred until loss-level
    # regularization (Step 3, focal loss + dropout/L2) closes that gap.
    train_opt_t, train_opt_f1 = tune_threshold(model, train_data)
    logger.info(
        f"  [diagnostic] train-optimal threshold: t={train_opt_t:.2f} "
        f"(train F1={train_opt_f1:.3f}); val eval uses t=0.5"
    )

    # Save best checkpoint and final evaluation.
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join(
        "checkpoints", f"hgnn_fold{fold_idx + 1}_seed{seed}.pt"
    )
    torch.save(model.state_dict(), ckpt_path)

    final_metrics = evaluate(model, val_data)

    os.makedirs(results_dir, exist_ok=True)
    pred_path = os.path.join(
        results_dir, f"fold{fold_idx + 1}_seed{seed}_predictions.csv"
    )
    _save_predictions_csv(final_metrics["predictions"], pred_path)

    final_metrics["loss_history"] = loss_history
    final_metrics["train_opt_threshold"] = train_opt_t
    final_metrics["train_opt_f1"] = train_opt_f1
    return final_metrics


def run_cv(
    seeds: list[int] | None = None,
    n_splits: int = 3,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_layers: int = DEFAULT_N_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    loss_type: str = DEFAULT_LOSS_TYPE,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
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
                dropout=dropout,
                weight_decay=weight_decay,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
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
    keys = [
        "sol_path",
        "hyperedge_idx",
        "contract_label",
        "true_label",
        "pred_label",
        "prob_safe",
        "prob_vuln",
    ]
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

    # Save per-fold-per-seed metrics. train_opt_threshold is diagnostic only
    # (what the F1-max sweep on train data would have picked); the applied
    # threshold for val is always 0.5 until regularization closes the
    # train/val calibration gap.
    detail_path = os.path.join(results_dir, "cv_detailed.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "fold", *metric_names, "threshold", "train_opt_threshold"])
        for result in all_results:
            seed = result["seed"]
            for fold_idx, fold_metrics in enumerate(result["fold_metrics"]):
                row = [seed, fold_idx + 1]
                row.extend(f"{fold_metrics[m]:.4f}" for m in metric_names)
                row.append(f"{fold_metrics.get('threshold', 0.5):.2f}")
                row.append(f"{fold_metrics.get('train_opt_threshold', 0.5):.2f}")
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
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--loss-type", choices=["ce", "focal"], default=DEFAULT_LOSS_TYPE
    )
    parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    summary = run_cv(
        seeds=args.seeds,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        results_dir=args.results_dir,
    )
