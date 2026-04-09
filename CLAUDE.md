# CLAUDE.md — Project Instructions for Claude Code

This file governs how Claude Code should behave throughout this project.
Read it fully before writing any code. Re-read it at the start of every new session.

---

## Project Overview

Build a static analysis system that detects **reentrancy vulnerabilities** in Solidity smart contracts using **transaction-centric hypergraph modeling** and a **Hypergraph Neural Network (HGNN)**.

The full specification is in `Hypergraph_Analysis.tex`. That file is the single source of truth for all definitions, formulas, and design decisions. When in doubt, refer to the spec — do not invent alternatives.

---

## Notation Rules

The spec uses precise mathematical notation. Variable names in code must reflect it directly.

| Spec Symbol | Code Name | Type | Description |
|---|---|---|---|
| $\mathcal{V}$ | `V` | `list` | Full node set |
| $\mathcal{V}_f$ | `V_f` | `list` | Function nodes |
| $\mathcal{V}_s$ | `V_s` | `list` | State variable nodes |
| $\mathcal{V}_c$ | `V_c` | `list` | External call site nodes |
| $\mathcal{E}$ | `E` | `list` | Hyperedge set |
| $\mathcal{H}$ | `H_graph` | `dict` | Hypergraph object `{V, E, X}` |
| $\mathbf{H}$ | `H_inc` | `np.ndarray` | Incidence matrix `|V| x |E|` |
| $G_{call}$ | `G_call` | `nx.DiGraph` | Call graph over V_f |
| $G_{dep}$ | `G_dep` | `nx.DiGraph` | Data dependency graph |
| $X$ | `X` | `np.ndarray` | Node feature matrix `|V| x d` |
| $x_v$ | `x_v` | `np.ndarray` | Feature vector for node v |
| $Z$ | `Z` | `np.ndarray` | Node embedding after L layers |
| $z_e$ | `z_e` | `np.ndarray` | Hyperedge embedding (mean pool) |
| $\hat{y}_e$ | `y_pred_e` | `int` | Predicted label for hyperedge e |
| $y_e$ | `y_true_e` | `int` | Ground truth label (1=vulnerable) |
| $e_c$ | `e_c` | `set` | Hyperedge for call site c |
| $F(c)$ | `F_c` | `set` | Call chain for call site c |
| $S(c)$ | `S_c` | `set` | Dependent state vars for call site c |

**Rules:**
- Never rename these variables to something more "Pythonic". Consistency with the spec is more important.
- `H_inc` and `H_graph` must never be confused — they are completely different objects.
- Node sets `V_f`, `V_s`, `V_c` must always be **disjoint**. Add an assertion to verify this wherever they are constructed.

---

## Architecture Constraints

Follow these exactly as specified in `Hypergraph_Analysis.tex`:

1. **One hyperedge per external call site** — `|E| == |V_c|` always.
2. **Hyperedge formula:** `e_c = {c} ∪ F_c ∪ S_c` (Section 4.5 of spec).
3. **HGNN propagation uses residual connections** — the base formula without residual is for reference only. Always implement the residual version.
4. **Hyperedge embedding = mean pooling** over member node embeddings — no attention, no max pooling.
5. **Classifier = linear layer + softmax** — output shape is `[2]` per hyperedge (non-vulnerable, vulnerable).
6. **Loss = cross-entropy** with class weighting to handle 1:2.6 imbalance (122 reentrant / 314 safe).
7. **L ∈ {2, 3, 4}** layers — make this a configurable hyperparameter, default = 2.

---

## Dataset

**Primary (train + validate):** `reentrancy-detection-benchmarks`
- URL: https://github.com/matteo-rizzo/reentrancy-detection-benchmarks
- Aggregated Benchmark: 436 contracts (122 reentrant / 314 safe) in `/benchmarks/aggregated-benchmark/`
- RS subset: 154 contracts in `/benchmarks/rs/` — used for ablation only
- Predefined CV splits: `cv_splits.zip` — always use these, do not generate custom splits
- AST generation: `scripts/source2ast.sh` (uses `solc`)
- CFG generation: `scripts/source2cfg.py` (uses Slither)

**Secondary (final held-out test only):** `manually-verified-reentrancy-dataset`
- URL: https://github.com/matteo-rizzo/manually-verified-reentrancy-dataset
- Never use this during training or ablation.
- Load it only in the final evaluation step (Step 8 of PLAN.md).

**Class imbalance:** use weighted cross-entropy. Weight for reentrant class = `314/122 ≈ 2.57`.

---

## Tech Stack

- **Language:** Python 3.10+
- **GNN framework:** PyTorch Geometric
- **Graph library:** NetworkX (`nx.DiGraph` for `G_call`, `G_dep`)
- **Solidity parsing:** use `solc` via subprocess for AST; Slither for CFG
- **Optimizer:** Adam, lr = 1e-3
- **Experiment tracking:** save metrics to CSV per run; no external tracking tools unless added later
- **Reproducibility:** set random seeds explicitly at the start of every training script; default seed = 42

---

## Code Style

- One file per step (matching PLAN.md step numbers), e.g., `step1_ast_cfg.py`, `step2_gdep.py`
- Every function must have a docstring referencing the spec section it implements, e.g.:
  ```python
  def build_hyperedge(c, G_call, G_dep, V_s):
      """
      Constructs hyperedge e_c for call site c.
      Spec: Section 4.5 — Hyperedge Definition.
      e_c = {c} ∪ F(c) ∪ S(c)
      """
  ```
- Add an assertion after every node set construction to verify disjointness.
- After each step, write a test using the reentrancy example from spec Section 4.7:
  ```solidity
  function withdraw(uint amount) {
      require(balance[msg.sender] >= amount);
      msg.sender.call{value: amount}("");
      balance[msg.sender] -= amount;
  }
  ```
  Expected hyperedge: `{withdraw(), balance[msg.sender], external_call_site}`

---

## Step Execution Rules

- **Complete one step fully before starting the next.** Do not partially implement a step.
- At the end of each step, confirm: (1) the module runs without error, (2) the worked example test passes, (3) output shapes match the spec.
- If anything in the spec is ambiguous, ask before assuming.
- Do not add features not described in the spec (e.g., no attention pooling, no extra baselines) unless explicitly asked.

---

## What NOT to Do

- Do not use `torch_geometric.data.HeteroData` — the hypergraph structure is custom, built manually via `H_inc`.
- Do not auto-generate the train/test split — always load from `cv_splits.zip`.
- Do not run Slither or `solc` on the held-out test set (Repo 2) during any training step.
- Do not rename spec variables for style reasons.
- Do not skip the worked example test at the end of each step.
