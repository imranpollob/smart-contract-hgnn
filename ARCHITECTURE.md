# ARCHITECTURE.md — Component Status Tracker

**Last updated:** 2026-04-16

---

## Component Status

| # | Component | File(s) | Status | Notes |
|---|-----------|---------|--------|-------|
| 0 | Project setup | `pyproject.toml`, dirs | ✅ Done | Using uv as package manager |
| 1 | AST, CFG, Call Graph extraction | `src/extraction/ast_cfg.py` | ✅ Done | 25 tests passing; uses Slither IR for call detection |
| 2 | Data Dependency Graph (G_dep) | `src/extraction/gdep.py` | ✅ Done | 12 tests passing; bipartite (call_site → state_var) |
| 3 | Node Set Construction (V_f, V_s, V_c) | `src/hypergraph/nodeset.py` | ✅ Done | 15 tests; disjointness asserted |
| 4 | Node Feature Matrix (X) | `src/hypergraph/features.py` | ✅ Done | d=34 (9 func + 14 var + 11 call); reentrancy flags/counts on V_s and V_c |
| 5 | Hyperedge Construction + Incidence Matrix (H) | `src/hypergraph/hyperedges.py` | ✅ Done | δ=3 BFS; 24 tests; binary H_inc verified |
| 6 | HGNN Model | `src/model/hgnn.py` | ✅ Done | 25 tests; softmax + CrossEntropyLoss; L∈{2,3,4} |
| 7 | Training Loop + CV Evaluation | `src/evaluation/train.py` | ✅ Done | 3-fold CV, 5 seeds, per-hyperedge labels (Slither detectors), per-fold class weights |
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
| 2026-04-09 | Step 0: uv init, installed deps, created src/ dir structure | pyproject.toml, src/*/__init__.py |
| 2026-04-09 | Step 1: AST/CFG/call graph extraction via Slither. Uses IR-level InternalCall for G_call edges. 25 tests. | src/extraction/ast_cfg.py, tests/test_extraction.py |
| 2026-04-09 | Step 2: G_dep bipartite graph. Reads before / writes after call via CFG node ordering. 12 tests. | src/extraction/gdep.py, tests/test_gdep.py |
| 2026-04-09 | Step 3: Node set construction. Fixed HighLevelCall .type bug in Step 1. 52 total tests, 50-contract pipeline 96% success. | src/hypergraph/nodeset.py, tests/test_nodeset.py, src/extraction/ast_cfg.py |
| 2026-04-09 | Step 4: Node feature matrix. d=26 (func:9, var:12, call:5). Fixed type classifier for structs. 77 total tests, 50-contract pipeline 0 code errors. | src/hypergraph/features.py, tests/test_features.py, feature_config.json |
| 2026-04-09 | Step 5: Hyperedge construction + H_inc. δ=3 bounded BFS. 101 total tests, 50-contract pipeline 0 code errors. | src/hypergraph/hyperedges.py, tests/test_hyperedges.py |
| 2026-04-10 | Step 6: HGNN model. Residual + LayerNorm, mean pooling, softmax classifier. 126 total tests. | src/model/hgnn.py, tests/test_hgnn.py |
| 2026-04-10 | Step 7: Training loop + CV evaluation. 3-fold stratified CV, 5-seed runs, weighted CE, metrics CSV. 146 total tests. | src/evaluation/train.py, tests/test_train.py |
| 2026-04-16 | Low F1 (~0.51) diagnosis + remediation. Root cause: contract-level labels smeared onto every hyperedge, and V_c features too coarse (5 bits). Fix: (a) per-call-site labels via Slither's reentrancy detectors in `src/extraction/labels.py`; (b) extended V_c with `gas_forwarded`, `sender_controlled_target`, `guarded_by_modifier` plus log1p counts of writes/reads around each call; (c) extended V_s with `written_after_call` / `read_before_call` aggregated from call sites; (d) dynamic per-fold class weights (`compute_class_weights`, clamp=10) from the actual per-hyperedge distribution — the legacy 1:2.57 contract-level ratio no longer matches the true positive rate. Feature dim 26 → 34. Fallback to contract label when Slither misses flagged calls in a reentrant contract. | src/extraction/labels.py (new), src/extraction/ast_cfg.py, src/hypergraph/features.py, src/evaluation/train.py, feature_config.json, notebooks/train_hgnn.ipynb, tests/test_labels.py (new), tests/test_features.py, tests/test_extraction.py, tests/test_train.py, CLAUDE.md, PLAN.md, README.md, docs/pipeline_walkthrough.md |

---

## Design Decisions

| # | Decision | Reason | Date |
|---|----------|--------|------|
| 1 | Softmax over 2 classes (not sigmoid) | Matches paper Section 5.5 explicitly. Equivalent for binary case but consistent with spec. | 2026-04-09 |
| 2 | δ=3 bounded BFS (not unbounded nx.ancestors) | CLAUDE.md specifies delta=3; unbounded traversal would include irrelevant distant ancestors. | 2026-04-09 |
| 3 | Generate CV splits via KFold (no cv_splits.zip) | cv_splits.zip not present in repo. Replicate make_folds.py logic for reproducibility. | 2026-04-09 |
| 4 | Secondary dataset labels from directory names | always-safe/ → 0, other categories → 1. No separate label file exists. | 2026-04-09 |
| 5 | Added baselines step (7.5) | Paper Section 7.2 requires GCN/GAT/MLP/RF/tool comparisons. Original PLAN omitted this. | 2026-04-09 |
| 6 | Per-call-site labels via Slither detectors instead of directory-level contract labels | Contract-level smearing forced every hyperedge in a reentrant contract to y=1, injecting ~60-80% label noise per contract (only 1-2 of the ~3-8 calls are actually vulnerable). Net effect: train.py cross-entropy chased a noisy target, F1 capped near 0.51. Slither's reentrancy detectors already localize the vulnerable call site — reuse their verdict. Fall back to the directory label if Slither flags nothing on a reentrant contract to avoid losing true positives. | 2026-04-16 |
| 7 | Reentrancy-specific features on V_c and V_s, dim 26 → 34 | The original V_c had 5 bits (opcode + value), which is nowhere near enough signal for a model to separate vulnerable calls from benign ones. The defining reentrancy pattern — external call that forwards gas → attacker-controlled target → state write after → no guard — is now encoded directly, giving the HGNN a usable gradient. | 2026-04-16 |
| 8 | Class weights computed per-fold from per-hyperedge distribution (clamp=10) | Once labels are per call site, the contract-level 1:2.57 ratio undershoots heavily — the actual positive rate is closer to 1:10. Dynamic computation tracks the real distribution; clamping stops the weight from blowing up on folds where a detector flags very few sites. | 2026-04-16 |
