# Smart Contract HGNN

Transaction-centric hypergraph modeling for reentrancy vulnerability detection in Solidity smart contracts using a Hypergraph Neural Network (HGNN).

## Overview

This system builds a **hypergraph** from a Solidity contract's program representation (AST, CFG, call graph, data dependencies) and classifies each external call site as vulnerable or non-vulnerable using an HGNN. Each hyperedge corresponds to one external call site and groups the call site node with its related function nodes (via call chain) and state variable nodes (via data dependency).

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)
- `solc` (Solidity compiler) — managed via `solc-select`

## Setup

```bash
# Clone the repo with submodules (datasets) — shallow clone for speed
git clone --recurse-submodules --depth 1 <repo-url>
cd smart-contract-hgnn

# If you've already cloned without submodules, initialize them:
git submodule update --init --recursive --depth 1

# Install dependencies with uv
uv sync

# Install solc versions (needed for compiling contracts)
uv run solc-select install 0.4.24 0.4.25 0.4.26 0.5.0 0.5.16 0.6.0 0.6.12 0.7.0 0.8.0 0.8.17
uv run solc-select use 0.8.0
```

## Dataset

The project uses two external datasets as **Git submodules**:

| Dataset                                                                                                      | Purpose                               | Location                                     |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------- | -------------------------------------------- |
| [reentrancy-detection-benchmarks](https://github.com/matteo-rizzo/reentrancy-detection-benchmarks)           | Training + Validation (436 contracts) | `data/reentrancy-detection-benchmarks/`      |
| [manually-verified-reentrancy-dataset](https://github.com/matteo-rizzo/manually-verified-reentrancy-dataset) | Held-out test set (Step 8 only)       | `data/manually-verified-reentrancy-dataset/` |

Datasets are cloned automatically with `--recurse-submodules`. To pull latest updates:

```bash
git submodule update --remote  # Update all submodules to latest
```

**Primary dataset structure:**
```
data/reentrancy-detection-benchmarks/benchmarks/aggregated-benchmark/src/
├── reentrant/   # 120 vulnerable contracts (label=1)
└── safe/        # 312 safe contracts (label=0)
```

## Project Structure

```
src/
├── extraction/
│   ├── ast_cfg.py      # Step 1: AST, CFG, call graph extraction via Slither
│   └── gdep.py         # Step 2: Data dependency graph (G_dep)
├── hypergraph/
│   ├── nodeset.py      # Step 3: Node set construction (V_f, V_s, V_c)
│   ├── features.py     # Step 4: Node feature matrix (X), d=26
│   └── hyperedges.py   # Step 5: Hyperedge construction + incidence matrix (H)
├── model/
│   └── hgnn.py         # Step 6: HGNN model (PyTorch)
└── evaluation/
    └── train.py        # Step 7: Training loop + 3-fold CV evaluation
```

## Running Tests

```bash
# Run all tests (146 tests)
uv run python -m pytest tests/ -v

# Run tests for a specific step
uv run python -m pytest tests/test_hgnn.py -v
```

## Training

### Quick run (1 seed, fewer epochs)

```bash
uv run python -m src.evaluation.train --seeds 42 --epochs 30
```

### Full evaluation (5 seeds, as per paper)

```bash
uv run python -m src.evaluation.train --seeds 42 0 1 2 3 --epochs 50
```

### All training options

```bash
uv run python -m src.evaluation.train \
    --epochs 50 \
    --lr 1e-3 \
    --hidden-dim 64 \
    --n-layers 2 \
    --seeds 42 0 1 2 3 \
    --results-dir results
```

| Flag            | Default    | Description                                        |
| --------------- | ---------- | -------------------------------------------------- |
| `--epochs`      | 50         | Training epochs per fold                           |
| `--lr`          | 1e-3       | Learning rate (Adam)                               |
| `--hidden-dim`  | 64         | HGNN hidden dimension                              |
| `--n-layers`    | 2          | Number of HGNN message passing layers (2, 3, or 4) |
| `--seeds`       | 42 0 1 2 3 | Random seeds for multi-run evaluation              |
| `--results-dir` | results    | Directory for output CSVs and metrics              |

### Output

Training produces:
- `results/cv_summary.csv` — mean and std of all metrics across seeds/folds
- `results/cv_detailed.csv` — per-seed, per-fold metrics
- `results/fold{N}_seed{S}_predictions.csv` — per-hyperedge predictions
- `checkpoints/hgnn_fold{N}_seed{S}.pt` — best model checkpoints

### Metrics reported

| Metric    | Description                           |
| --------- | ------------------------------------- |
| Precision | TP / (TP + FP) for vulnerable class   |
| Recall    | TP / (TP + FN) for vulnerable class   |
| F1        | Harmonic mean of precision and recall |
| FNR       | FN / (FN + TP) — false negative rate  |
| FPR       | FP / (FP + TN) — false positive rate  |

## Technical Details

- **HGNN architecture**: Spectral convolution on hypergraphs (Feng et al., 2019) with residual connections, LayerNorm, and mean pooling
- **Loss**: Weighted CrossEntropyLoss (class weights: safe=1.0, vulnerable=2.57 to handle 1:2.6 imbalance)
- **CV**: 3-fold stratified cross-validation via `sklearn.KFold(n_splits=3, shuffle=True, random_state=42)`, applied per class
- **Features**: 26-dimensional node features (function: 9, state variable: 12, call site: 5)
- **Hyperedges**: One per external call site, bounded ancestor expansion with delta=3
