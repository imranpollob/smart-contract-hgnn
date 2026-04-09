# PLAN.md — Step-by-Step Build Plan

Each step maps to a section in `Hypergraph_Analysis.tex`.
Complete steps in order. Do not skip ahead.
Each step ends with a verification checklist — all items must pass before moving on.

---

## Step 0 — Project Setup
**Output:** Directory structure, `requirements.txt`, `ARCHITECTURE.md`

### Task
Create the project skeleton and install dependencies.

### Directory Structure
```
smart-contract-hgnn/
├── CLAUDE.md
├── PLAN.md
├── ARCHITECTURE.md
├── Hypergraph_Analysis.tex
├── requirements.txt
├── data/
│   ├── reentrancy-detection-benchmarks/   ← Repo 1 (already cloned)
│   └── manually-verified-reentrancy-dataset/  ← Repo 2 (already cloned)
├── src/
│   ├── __init__.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── ast_cfg.py          ← Step 1
│   │   └── gdep.py             ← Step 2
│   ├── hypergraph/
│   │   ├── __init__.py
│   │   ├── nodeset.py          ← Step 3
│   │   ├── features.py         ← Step 4
│   │   └── hyperedges.py       ← Step 5
│   ├── model/
│   │   ├── __init__.py
│   │   └── hgnn.py             ← Step 6
│   └── evaluation/
│       ├── __init__.py
│       ├── train.py            ← Step 7
│       ├── baselines.py        ← Step 7.5
│       ├── final_eval.py       ← Step 8
│       ├── ablation.py         ← Step 9
│       └── case_study.py       ← Step 10
├── scripts/
├── tests/
│   ├── __init__.py
│   ├── test_extraction.py
│   ├── test_gdep.py
│   ├── test_nodeset.py
│   ├── test_features.py
│   ├── test_hyperedges.py
│   └── test_hgnn.py
├── checkpoints/
├── results/
└── feature_config.json
```

### Verification Checklist
- [ ] All directories exist
- [ ] `requirements.txt` lists all dependencies
- [ ] `ARCHITECTURE.md` created with all components at ⏳ Pending

---

## Step 1 — AST, CFG, Call Graph Extraction
**Spec reference:** Section 3.2 (Program Representation Layer)
**Output file:** `src/extraction/ast_cfg.py`
**Test file:** `tests/test_extraction.py`

### Task
Parse a Solidity contract and extract:
- AST using `solc --ast-compact-json` via subprocess
- CFG using Slither
- Call graph `G_call` as `nx.DiGraph` where edge `f_i → f_j` means `f_i` calls `f_j`

### Inputs
- Path to a `.sol` Solidity source file

### Outputs
- `ast: dict` — raw AST JSON from solc
- `G_call: nx.DiGraph` — nodes = function names, edges = call relationships
- `cfg: dict` — CFG per function (Slither output)

### Implementation Notes
- Use `solc-select` to match the pragma version of each contract automatically
- `G_call` nodes must match the function names that will become `V_f` in Step 3
- Store `G_call` as a NetworkX DiGraph — nodes are string function names
- Wrap all Slither/solc calls in try-except; log failures, do not crash the pipeline

### Verification Checklist
- [ ] AST parses successfully for the `withdraw` example contract
- [ ] `G_call` is a valid `nx.DiGraph` with at least one node per function
- [ ] No crash on contracts with no external calls
- [ ] No crash on contracts with multiple pragma versions

---

## Step 2 — Data Dependency Graph ($G_{dep}$)
**Spec reference:** Section 3.2 and Section 4.4.2
**Output file:** `src/extraction/gdep.py`
**Test file:** `tests/test_gdep.py`

### Task
Build `G_dep: nx.DiGraph` — a bipartite graph where edge `(n, s)` exists if state variable `s` is read before or written after node `n` (where `n` is a call site or function).

### Inputs
- AST from Step 1
- CFG from Step 1

### Outputs
- `G_dep: nx.DiGraph` — bipartite; left nodes = call sites/functions, right nodes = state variables

### Implementation Notes
- "Read before `c`" means: `s` appears in a read position in any statement that executes before the external call `c` within the same function body
- "Written after `c`" means: `s` appears in a write (assignment) position in any statement that executes after `c` within the same function body
- Use the AST to identify read/write positions statically

### Verification Checklist
- [ ] For `withdraw` example: `(external_call, balance[msg.sender])` is in `G_dep`
- [ ] `G_dep` is a `nx.DiGraph`
- [ ] No state variable appears as both a left-side and right-side node

---

## Step 3 — Node Set Construction ($\mathcal{V}$)
**Spec reference:** Section 4.2 (Node Set)
**Output file:** `src/hypergraph/nodeset.py`
**Test file:** `tests/test_nodeset.py`

### Task
Build the three disjoint node sets and combine into `V`.

### Inputs
- AST, `G_call`, `G_dep` from Steps 1–2

### Outputs
- `V_f: list` — function nodes
- `V_s: list` — state variable nodes
- `V_c: list` — external call site nodes
- `V: list` — combined node set `V_f + V_s + V_c`
- `node_index: dict` — maps each node identifier to its integer index in `V`

### Implementation Notes
- Each node is a unique string identifier:
  - Functions: `"func:<function_name>"`
  - State variables: `"var:<variable_name>"`
  - Call sites: `"call:<function_name>:<line_number>"`
- `node_index` is used to build `H_inc` in Step 5

### Verification Checklist
- [ ] `set(V_f) ∩ set(V_s) == set()` — assertion in code
- [ ] `set(V_f) ∩ set(V_c) == set()` — assertion in code
- [ ] `set(V_s) ∩ set(V_c) == set()` — assertion in code
- [ ] For `withdraw` example: `V_c` contains exactly one call site node
- [ ] `len(V) == len(V_f) + len(V_s) + len(V_c)`

---

## Step 4 — Node Feature Matrix ($X$)
**Spec reference:** Section 4.3 (Node Features)
**Output file:** `src/hypergraph/features.py`
**Test file:** `tests/test_features.py`

### Task
Compute feature vector `x_v ∈ R^d` for each node `v ∈ V` and assemble into matrix `X`.

### Inputs
- `V_f`, `V_s`, `V_c`, `V` from Step 3
- AST from Step 1

### Outputs
- `X: np.ndarray` of shape `(|V|, d)` — node feature matrix

### Feature Definitions (per node type)
| Node type | Features |
|---|---|
| `V_f` (functions) | AST node type (one-hot), visibility (`public/private/internal/external`, one-hot), mutability (`pure/view/payable/nonpayable`, one-hot) |
| `V_s` (state vars) | Variable type (one-hot over common Solidity types), storage slot index (normalized), access pattern (read-only / write-only / read-write, one-hot) |
| `V_c` (call sites) | Call opcode (CALL / DELEGATECALL / STATICCALL, one-hot), value-transfer flag (0 or 1) |

- Pad all feature vectors to the same dimension `d` with zeros.
- Record `d` and the feature schema in a `feature_config.json` file for reproducibility.

### Verification Checklist
- [ ] `X.shape == (len(V), d)`
- [ ] No NaN or Inf values in `X`
- [ ] All rows for `V_f` nodes have zero padding in the `V_s` and `V_c` feature slots
- [ ] `feature_config.json` is written

---

## Step 5 — Hyperedge Construction and Incidence Matrix ($\mathbf{H}$)
**Spec reference:** Section 4.5–4.6 (Hyperedge Definition + Algorithm)
**Output file:** `src/hypergraph/hyperedges.py`
**Test file:** `tests/test_hyperedges.py`

### Task
For each call site `c ∈ V_c`, construct hyperedge `e_c = {c} ∪ F(c) ∪ S(c)`.
Then build incidence matrix `H_inc`.

### Inputs
- `V`, `V_c`, `V_s`, `node_index` from Step 3
- `G_call`, `G_dep` from Steps 1–2

### Outputs
- `E: list[set]` — list of hyperedges (each is a set of node identifiers)
- `H_inc: np.ndarray` of shape `(|V|, |E|)` — binary incidence matrix

### Implementation Notes
- `F(c) = {f} ∪ BoundedAncestors(f, G_call, delta)` where `f` is the function containing `c`
  - **Use BFS with depth cutoff δ=3** (not unbounded `nx.ancestors()`). Implement as:
    ```python
    def bounded_ancestors(G, node, delta=3):
        """Return ancestors of node reachable within delta hops via BFS."""
        visited = set()
        frontier = {node}
        for _ in range(delta):
            next_frontier = set()
            for n in frontier:
                for pred in G.predecessors(n):
                    if pred not in visited:
                        visited.add(pred)
                        next_frontier.add(pred)
            frontier = next_frontier
        return visited
    ```
- `S(c) = {s ∈ V_s | (c, s) ∈ G_dep}`
- `H_inc[i, j] = 1` if node `V[i]` is in hyperedge `E[j]`, else 0
- Assert `len(E) == len(V_c)` — one hyperedge per call site

### Verification Checklist
- [ ] For `withdraw` example: `e_c` contains `withdraw`, `balance[msg.sender]`, and the call site node
- [ ] `H_inc.shape == (len(V), len(V_c))`
- [ ] `H_inc` contains only 0s and 1s
- [ ] `len(E) == len(V_c)` — assertion in code
- [ ] Every call site node `c` appears in exactly its own hyperedge
- [ ] Ancestor expansion respects δ=3 bound

---

## Step 6 — HGNN Model
**Spec reference:** Section 5 (Hypergraph Neural Network)
**Output file:** `src/model/hgnn.py`
**Test file:** `tests/test_hgnn.py`

### Task
Implement the HGNN as a PyTorch `nn.Module`.

### Inputs (at forward pass)
- `X: torch.Tensor` shape `(|V|, d)` — node features
- `H_inc: torch.Tensor` shape `(|V|, |E|)` — incidence matrix

### Outputs (at forward pass)
- `y_pred: torch.Tensor` shape `(|E|, 2)` — softmax probabilities per hyperedge

### Architecture (implement exactly)

```
1. Compute D_v (diagonal, shape |V|x|V|): D_v[i,i] = sum over e of H_inc[i,e]
2. Compute D_e (diagonal, shape |E|x|E|): D_e[e,e] = sum over v of H_inc[v,e]
3. Compute H_tilde = D_v^{-1/2} @ H_inc @ D_e^{-1/2}
4. For l in range(L):
       X_new = sigma(H_tilde @ H_tilde.T @ X @ W[l])
       X = X + X_new                          # residual connection
       X = LayerNorm(X)                        # optional, configurable
5. Z = X                                       # final node embeddings, shape (|V|, d')
6. For each hyperedge e_j in E:
       z_e = mean(Z[v] for v in e_j)          # mean pooling
7. y_pred = softmax(z_e @ W_c.T + b_c)        # classifier
```

### Implementation Notes
- `W[l]` is a learnable `nn.Linear(d, d)` per layer (no bias in the propagation step)
- `W_c` is a learnable `nn.Linear(d', 2)` classifier head
- `L` is a constructor argument, default = 2
- LayerNorm is a constructor argument `use_layernorm: bool`, default = True
- Handle `D_v[i,i] = 0` (isolated nodes) by setting inverse to 0
- **Classifier uses softmax** (per paper Section 5.5): output is `[p(non-vulnerable), p(vulnerable)]`
- **Loss: `nn.CrossEntropyLoss`** with class weights (not BCE)

### Verification Checklist
- [ ] Forward pass runs without error on the `withdraw` example
- [ ] Output shape is `(|E|, 2)` where `|E| == |V_c|`
- [ ] Output rows sum to 1.0 (valid probability distribution)
- [ ] Changing `L` changes the number of parameters correctly
- [ ] Gradient flows back through all layers (check with `loss.backward()`)

---

## Step 7 — Training Loop and CV Evaluation
**Spec reference:** Section 7 (Evaluation Plan)
**Output file:** `src/evaluation/train.py`

### Task
Implement the full training loop using 3-fold CV splits.

### CV Split Generation
There is no pre-existing `cv_splits.zip`. Generate splits programmatically using the same logic
as `data/reentrancy-detection-benchmarks/scripts/make_folds.py`:
```python
from sklearn.model_selection import KFold

# Collect contract filenames from:
#   data/reentrancy-detection-benchmarks/benchmarks/aggregated-benchmark/src/reentrant/
#   data/reentrancy-detection-benchmarks/benchmarks/aggregated-benchmark/src/safe/
# Sort filenames for reproducibility.
# Apply KFold(n_splits=3, shuffle=True, random_state=42) per class.
```

### Inputs
- Aggregated Benchmark contracts (436 contracts) from Repo 1
- CV splits generated as above

### Per-Fold Loop
```
for fold in [1, 2, 3]:
    generate train/val contract lists from KFold split
    for each contract: run Steps 1-5 to build H_graph, H_inc, X, E, labels
    train HGNN for N epochs:
        forward pass → y_pred
        compute weighted cross-entropy loss
        backward + Adam step
    evaluate on val set → Precision, Recall, F1, FNR, FPR
save metrics to results/fold_{fold}_metrics.csv
```

### Label Assignment
Labels are at the contract level (directory: `reentrant/` vs `safe/`).
Map to hyperedge labels using the rule from spec Section 6.2:
- For contracts in `reentrant/`: ALL hyperedges get `y_e = 1`
- For contracts in `safe/`: ALL hyperedges get `y_e = 0`

### Class Weights
- Reentrant (class 1) weight = `314 / 122 ≈ 2.57`
- Safe (class 0) weight = `1.0`
- Pass as `weight` argument to `nn.CrossEntropyLoss`

### Metrics to Compute and Save
- Precision (vulnerable class)
- Recall (vulnerable class)
- F1-score (vulnerable class)
- FNR = FN / (FN + TP)
- FPR = FP / (FP + TN)
- Per-hyperedge predictions saved to CSV for localization analysis

### Reproducibility
- Set `torch.manual_seed(42)` and `np.random.seed(42)` at start
- Run 5 seeds: `[42, 0, 1, 2, 3]`; report mean ± std across seeds

### Verification Checklist
- [ ] Training loss decreases over epochs
- [ ] CV splits are generated deterministically (same filenames each run)
- [ ] Metrics saved to `results/` per fold
- [ ] 5-seed run completes and mean ± std is reported

---

## Step 7.5 — Baselines
**Spec reference:** Section 7.2 (Baselines)
**Output file:** `src/evaluation/baselines.py`

### Task
Implement baseline comparisons to validate that the HGNN hypergraph approach adds value.

### Baselines to Implement

| Baseline | Description |
|---|---|
| **GCN** | Replace `H_inc` with a pairwise adjacency matrix derived from hyperedges (clique expansion). Same node features, same classifier head. Use PyTorch Geometric `GCNConv`. |
| **GAT** | Same as GCN but with `GATConv`. |
| **MLP** | Flatten hyperedge features (mean of member node features) → 2-layer MLP classifier. |
| **Random Forest** | Same flattened features → sklearn `RandomForestClassifier`. |
| **Static analysis tools** | Load pre-computed results from `data/reentrancy-detection-benchmarks/results/smartbugs/*.csv`. Parse tool predictions and compute metrics. |

### Implementation Notes
- GCN/GAT: use the same training loop, optimizer, and CV splits as Step 7
- MLP/RF: use the same CV splits; train sklearn models on the same features
- Tool baselines: parse existing CSV results, no re-running tools needed

### Verification Checklist
- [ ] All 5 baselines produce metrics on the same CV splits
- [ ] Results saved to `results/baselines_*.csv`

---

## Step 8 — Final Held-Out Evaluation (Repo 2)
**Spec reference:** Section 7.1 (Secondary Dataset)
**Output file:** `src/evaluation/final_eval.py`

### Task
Evaluate the best trained model (highest val F1 across folds/seeds) on the held-out test set from `manually-verified-reentrancy-dataset`.

### Label Mapping for Secondary Dataset
The secondary dataset organizes contracts by category, not binary labels.
Map directory structure to labels:

```
dataset/0_8/always-safe/*           → label 0
dataset/0_8/cross-contract/*        → label 1
dataset/0_8/cross-function/*        → label 1
dataset/0_8/single-function/*       → label 1

dataset/0_4/always-safe/*           → label 0
dataset/0_4/{other categories}/*    → label 1

dataset/0_5/always-safe/*           → label 0
dataset/0_5/{other categories}/*    → label 1
```

### Rules
- Load weights from the best checkpoint saved in Step 7
- Do NOT retrain or fine-tune on Repo 2 data
- Run the same preprocessing pipeline (Steps 1–5) on Repo 2 contracts

### Outputs
- `results/final_eval_metrics.csv` — same metric columns as Step 7
- `results/final_eval_predictions.csv` — per-hyperedge predictions with contract name and source line mapping

### Verification Checklist
- [ ] Model is loaded from checkpoint, not retrained
- [ ] Label mapping from directory categories is correct
- [ ] Metrics are saved to `results/`
- [ ] Predictions are mapped back to source lines for localization

---

## Step 9 — Ablation Studies
**Spec reference:** Section 7.4 (Ablation Study)
**Output file:** `src/evaluation/ablation.py`

Run 5 controlled ablations, each as a separate training run on the Aggregated Benchmark (same CV splits):

| Ablation | What to change |
|---|---|
| A1: Hypergraph → Graph | Replace `H_inc` with a standard pairwise adjacency matrix (clique expansion of hyperedges) |
| A2: Remove S(c) | Set `S_c = set()` in Step 5; rebuild `H_inc` |
| A3: Remove F(c) ancestors | Set `F_c = {f}` only (no ancestor expansion); rebuild `H_inc` |
| A4: No residual connections | Remove `X = X + X_new` in HGNN forward pass |
| A5: Depth sensitivity | Train with `L ∈ {1, 2, 3, 4}`; plot F1 vs L |

Save all ablation results to `results/ablation_*.csv`.

### Verification Checklist
- [ ] All 5 ablations complete without error
- [ ] Results saved to `results/`
- [ ] A5 produces 4 rows (one per L value)

---

## Step 10 — Case Study and Localization
**Spec reference:** Section 7.5 (Case Study)
**Output file:** `src/evaluation/case_study.py`

### Task
Select one contract from Repo 2 with a confirmed reentrancy vulnerability. For the predicted vulnerable hyperedge, show:
1. All member nodes in `e_c` with their types (`V_f`, `V_s`, or `V_c`)
2. Predicted label `y_pred_e` and confidence score `p_e`
3. Mapping of each node back to its source code line number

### Output
- Print a human-readable report to stdout
- Save to `results/case_study.txt`

### Verification Checklist
- [ ] At least one contract with a vulnerable prediction is shown
- [ ] Source line mapping is accurate (verify against the `.sol` file manually)
- [ ] Report is saved to `results/case_study.txt`
