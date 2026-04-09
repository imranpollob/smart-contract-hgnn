# ARCHITECTURE.md — Component Status Tracker

**Last updated:** 2026-04-09

---

## Component Status

| # | Component | File(s) | Status | Notes |
|---|-----------|---------|--------|-------|
| 0 | Project setup | `requirements.txt`, dirs | ⏳ Pending | |
| 1 | AST, CFG, Call Graph extraction | `src/extraction/ast_cfg.py` | ⏳ Pending | |
| 2 | Data Dependency Graph (G_dep) | `src/extraction/gdep.py` | ⏳ Pending | |
| 3 | Node Set Construction (V_f, V_s, V_c) | `src/hypergraph/nodeset.py` | ⏳ Pending | |
| 4 | Node Feature Matrix (X) | `src/hypergraph/features.py` | ⏳ Pending | |
| 5 | Hyperedge Construction + Incidence Matrix (H) | `src/hypergraph/hyperedges.py` | ⏳ Pending | δ=3 depth bound |
| 6 | HGNN Model | `src/model/hgnn.py` | ⏳ Pending | softmax + CrossEntropyLoss |
| 7 | Training Loop + CV Evaluation | `src/evaluation/train.py` | ⏳ Pending | KFold(3, shuffle, seed=42) |
| 7.5 | Baselines (GCN, GAT, MLP, RF, tools) | `src/evaluation/baselines.py` | ⏳ Pending | |
| 8 | Final Held-Out Evaluation (Repo 2) | `src/evaluation/final_eval.py` | ⏳ Pending | Label mapping from dirs |
| 9 | Ablation Studies (A1–A5) | `src/evaluation/ablation.py` | ⏳ Pending | |
| 10 | Case Study + Localization | `src/evaluation/case_study.py` | ⏳ Pending | |

---

## Known Issues

| # | Component | Issue | Date Logged | Status |
|---|-----------|-------|-------------|--------|
| — | — | No issues yet | — | — |

---

## Implementation Log

| Date | What was done | Files touched |
|------|---------------|---------------|
| 2026-04-09 | Project review: verified feasibility, revised PLAN.md (7 fixes), updated CLAUDE.md, created ARCHITECTURE.md | PLAN.md, CLAUDE.md, ARCHITECTURE.md |

---

## Design Decisions

| # | Decision | Reason | Date |
|---|----------|--------|------|
| 1 | Softmax over 2 classes (not sigmoid) | Matches paper Section 5.5 explicitly. Equivalent for binary case but consistent with spec. | 2026-04-09 |
| 2 | δ=3 bounded BFS (not unbounded nx.ancestors) | CLAUDE.md specifies delta=3; unbounded traversal would include irrelevant distant ancestors. | 2026-04-09 |
| 3 | Generate CV splits via KFold (no cv_splits.zip) | cv_splits.zip not present in repo. Replicate make_folds.py logic for reproducibility. | 2026-04-09 |
| 4 | Secondary dataset labels from directory names | always-safe/ → 0, other categories → 1. No separate label file exists. | 2026-04-09 |
| 5 | Added baselines step (7.5) | Paper Section 7.2 requires GCN/GAT/MLP/RF/tool comparisons. Original PLAN omitted this. | 2026-04-09 |
