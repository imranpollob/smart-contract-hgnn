# Weekly Update — Transaction-Centric Hypergraph for Reentrancy Detection

**Date:** 2026-04-24
**Presenter:** Imran
**Audience:** Advisor + lab members (mixed background: some know static analysis, some know GNNs, some know neither)
**Format:** ~10 min walkthrough, ~20 min brainstorming

This deck is written as if I were explaining the project to a new teammate joining today. Each slide has the "what" and the "why" in full sentences — nothing is left as a one-liner.

---

## Slide 1 — One-minute recap

**What the project does.** Given a Solidity smart contract, we want to automatically decide whether any of its *external calls* (calls that hand control over to another contract, like `msg.sender.call{value: ...}("")`) can be exploited by the reentrancy attack pattern. Reentrancy is the vulnerability class that caused the 2016 DAO hack (~$60M stolen) and continues to cause real losses today — a contract calls an attacker's address, and the attacker re-enters the original contract *before its bookkeeping has finished*, draining funds.

**Why we don't just ask "is this contract vulnerable?"** Because a single contract usually has multiple external calls, and only one or two of them are actually the dangerous ones. A contract-level yes/no answer is useless to an auditor — they already know something *somewhere* is wrong; they need to know *which line*. So we classify at the **external-call-site level**: every external call gets its own prediction.

**Why a hypergraph and not a regular graph.** A reentrancy exploit is not a pair of nodes connected by an edge — it's a *pattern* that involves at least three things acting together: (1) the external call itself, (2) the function(s) that contain and can trigger it, and (3) the state variable(s) that are read before and written after that call. We bundle those three things into one *hyperedge* per call site, so the model sees the full exploit pattern as a single unit rather than trying to reassemble it from scattered pairwise edges.

**Where we were last week.** The full static pipeline was running, but F1 was stuck at ~0.51. We diagnosed the root cause: labels were assigned at the *contract* level, so every call site in a reentrant contract got `y=1` even when only one of several calls was actually vulnerable. That injected ~60–80% label noise per contract.

**What changed this week.** Three fixes landed together:
1. Labels are now **per call site**, derived from Slither's reentrancy detectors (they already localize which specific call is vulnerable — we just reuse their verdict).
2. Node features extended from 26 to 34 dimensions, adding reentrancy-pattern bits directly on state-variable and call-site nodes (e.g. `read_before_call`, `written_after_call`, `gas_forwarded`, `guarded_by_modifier`).
3. Class weights for the loss are now recomputed per fold from the real per-hyperedge positive/negative ratio, not from the old contract-level 1:2.57 ratio.

The first end-to-end CV run on the new pipeline finished this week — results on Slide 9.

---

## Slide 2 — System architecture (the four stages)

```
┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐   ┌──────────────────┐
│ 1. Extraction   │ → │ 2. Hypergraph    │ → │ 3. HGNN     │ → │ 4. Classify +    │
│ AST, CFG,       │   │ V = V_f ∪ V_s    │   │ L residual  │   │ localize         │
│ G_call, G_dep   │   │ ∪ V_c, build E   │   │ layers      │   │ hyperedge        │
│ (solc + Slither)│   │ and H_inc        │   │ + mean pool │   │ (softmax, CE)    │
└─────────────────┘   └──────────────────┘   └─────────────┘   └──────────────────┘
```

**Stage 1 — Extraction.** We never look at the compiled bytecode. We stay at the source level because we need to know *why* a call is dangerous (which state variables it touches), not just that it exists. We use two static tools:
- `solc` (the Solidity compiler) gives us the AST — the syntactic tree of the program.
- `Slither` gives us control-flow graphs per function, plus a call graph `G_call` that says "function f can call function g" inside the contract. We also build a data-dependency graph `G_dep` that says "call site c reads/writes state variable s."

**Stage 2 — Hypergraph construction.** From the four graphs above, we build one hypergraph per contract. Nodes are of three disjoint types: `V_f` (one node per function), `V_s` (one per state variable), `V_c` (one per external call site). Hyperedges are one per call site — more on the exact formula in Slide 5.

**Stage 3 — HGNN.** A hypergraph neural network runs L layers of message passing (default L=2). Each layer lets information flow from every node to every other node that shares a hyperedge with it. The output is an embedding per node.

**Stage 4 — Classification.** For each hyperedge, we average the embeddings of its member nodes to get a hyperedge embedding `z_e`, then a linear + softmax classifier gives us `[P(safe), P(vulnerable)]`.

**Status** (from [ARCHITECTURE.md](../ARCHITECTURE.md)): stages 1–3 are done with passing tests (~150 unit tests total). Stage 4 works but is the part currently under active tuning. Not yet done: baselines (GCN/GAT/MLP/RF/tool comparisons), held-out evaluation on the second dataset, ablation studies, and the localization case study.

---

## Slide 3 — Worked example: a real reentrant contract

```solidity
// simplified from the DAO-style pattern — this is the textbook reentrancy bug
contract Vault {
    mapping(address => uint) public balance;

    function withdraw(uint amount) public {
        require(balance[msg.sender] >= amount);     // (1) READ state
        msg.sender.call{value: amount}("");          // (2) EXTERNAL CALL (unguarded)
        balance[msg.sender] -= amount;               // (3) WRITE state AFTER the call
    }
}
```

**Why this is "the" reentrancy pattern.** Three things in order: (1) read a state variable that gates the operation, (2) make an external call that transfers control out of our contract, (3) write the state variable that was read in (1) — *after* the call. The attacker's contract is whatever receives the call in (2); it can call `withdraw` again before line (3) runs, so the `require` in (1) still passes on re-entry, and the attacker drains the vault.

**How a fix would look.** Either (a) reorder so the write happens before the call (the "checks-effects-interactions" pattern), or (b) wrap the function with a `nonReentrant` modifier that rejects recursive entry. Our model should fire on this contract and *stop* firing once either fix is applied.

**Why we use this example.** It is the simplest contract that contains the full pattern. Every pipeline step has a unit test that operates on (a slightly expanded version of) this code, so we can verify at each stage that the right nodes, hyperedges, and labels come out.

---

## Slide 4 — Example: what the extraction stage pulls out

From the `withdraw` contract above, we extract three kinds of nodes. Each gets a feature vector in `R^34`; zero-padding aligns them into one matrix.

**V_f (functions, 9-dim slice):** one node.
- `func:withdraw` → features = [visibility=public(one-hot), mutability=payable(one-hot), is_constructor=0, …]
- These features say *what kind of function* this is — not whether it's buggy. Visibility matters because `private` functions can't be called from outside, so they can never be the entry point of an attack.

**V_s (state variables, 14-dim slice):** one node.
- `var:balance` → features = [type=mapping(one-hot), storage_slot=0, access_pattern=read+write, **read_before_call=1, written_after_call=1**, …]
- The two **bold** bits are the reentrancy-specific flags we added this week. They are set by scanning the CFG of every function and asking: is this variable read on some path before reaching an external call, and written on some path after it? If both are true, this variable is part of the classic pattern.

**V_c (call sites, 11-dim slice):** one node.
- `call:withdraw:6` → features = [opcode=CALL(one-hot), value_transfer=1, **gas_forwarded=1, sender_controlled_target=1, guarded_by_modifier=0**, log1p(writes_after_call)=log(2), …]
- The **bold** bits again encode the reentrancy pattern directly. `gas_forwarded` is 1 because `.call{value: ...}` forwards all gas by default (enough gas to re-enter). `sender_controlled_target` is 1 because the target is `msg.sender` — the attacker. `guarded_by_modifier` is 0 because the function has no `nonReentrant` modifier. Together these three bits light up exactly on the vulnerable call.

**Why features matter so much here.** The HGNN message passing mixes information across nodes, but the *raw signal* has to come from somewhere. If the call-site node has no bit that says "this call is suspicious," the model has nothing to work with except topology, which is too weak. Moving from 26 → 34 features was specifically to inject per-node reentrancy evidence.

**Disjointness.** `V_f ∩ V_s = V_f ∩ V_c = V_s ∩ V_c = ∅`. Asserted in code at [nodeset.py](../src/hypergraph/nodeset.py) so this can never silently break.

---

## Slide 5 — Example: how the hyperedge is built

The hyperedge for call site `c` is:

```
e_c = {c} ∪ F(c) ∪ S(c)
```

Unpacking each piece on our example:

- **{c}** = `{call:withdraw:6}` — the call site itself.
- **F(c)** = `{withdraw} ∪ BoundedAncestors(withdraw, G_call, δ=3)`. The "ancestors" are functions that can reach `withdraw` through the call graph within 3 hops. We bound the depth because an unbounded expansion sucks in distant utility functions that have nothing to do with this exploit. Our toy contract has no callers of `withdraw`, so F(c) = `{withdraw}`.
- **S(c)** = every state variable `s` such that `(c, s) ∈ G_dep`. In our example, `balance` is read before the call (required by the `require`) and written after the call (the `-=` line), so S(c) = `{balance}`.

Putting it together:

```
e_c = { call:withdraw:6,  func:withdraw,  var:balance }
```

**The incidence matrix for this contract.** `|V|=3`, `|E|=1`:

```
           e_1
H_inc = [  1  ]    row 0: func:withdraw
        [  1  ]    row 1: var:balance
        [  1  ]    row 2: call:withdraw:6          shape = (3,1), binary
```

**Why this shape of hyperedge.** A hyperedge is supposed to capture one *transaction logic unit*. A reentrancy exploit is a transaction that reads some state, calls out, and then updates that state — so the minimal unit that contains the whole pattern is exactly these three things. Not the whole contract, not a single line.

**One practical consequence.** Because there is exactly one hyperedge per external call site, `|E| = |V_c|` always. We assert this. If the pipeline ever produces a different count, something is broken in extraction.

---

## Slide 6 — The HGNN: what the algorithm actually computes

**Inputs at forward time:** node feature matrix `X ∈ R^{|V|×d}`, binary incidence matrix `H_inc ∈ {0,1}^{|V|×|E|}`, plus the list of hyperedges `E` (used for the mean-pooling step).

**Step 1 — Normalize the incidence.** We compute two diagonal degree matrices:
- `D_v[i,i]` = how many hyperedges node `i` belongs to.
- `D_e[e,e]` = how many nodes hyperedge `e` contains.

Then:

$$\tilde H = D_v^{-1/2}\, H_{inc}\, D_e^{-1/2}$$

This is the hypergraph analogue of the symmetric normalization used in GCNs. It prevents high-degree nodes (popular functions, frequently-touched state variables) from dominating the signal purely because they have more connections.

**Step 2 — Propagate, L times.** The per-layer update is:

$$X^{(l+1)} = \mathrm{LayerNorm}\big(X^{(l)} + \sigma(\tilde H\, \tilde H^\top X^{(l)} W^{(l)})\big)$$

Reading it left-to-right: `$\tilde H \tilde H^\top$` is an `|V|×|V|` matrix where entry `(i,j)` is positive when nodes `i` and `j` share a hyperedge. Multiplying by `X` mixes features across co-members of any hyperedge. `W^{(l)}` is a learnable linear transform. `σ` is ReLU. The `+` is a **residual connection** — we *add* the mixed signal to the input rather than replacing it. This stabilizes training for deeper networks and prevents oversmoothing (every node collapsing to the same embedding). `LayerNorm` keeps activations in a reasonable range across layers.

**Step 3 — Pool per hyperedge.** After L layers, we have node embeddings `Z`. For each hyperedge `e`:

$$z_e = \frac{1}{|e|} \sum_{v \in e} Z_v$$

Plain **mean pool** — no attention, no max. We picked mean because every member of the hyperedge is load-bearing (drop any one and you've broken the reentrancy pattern), so the model shouldn't be free to ignore some members.

**Step 4 — Classify.**

$$\hat y_e = \mathrm{softmax}(W_c z_e + b_c) \quad\in\ R^2$$

Output = `[P(safe), P(vulnerable)]` per hyperedge.

Code + 25 unit tests: [src/model/hgnn.py](../src/model/hgnn.py).

**Default hyperparameters.** L=2 layers, hidden_dim=64, dropout=0.0 (about to raise), weight_decay=0.0 (about to raise), use_layernorm=True.

---

## Slide 7 — Loss function and class imbalance

**What we optimize.** Standard PyTorch `nn.CrossEntropyLoss` with class weights. Focal loss is implemented as a drop-in alternative in [src/model/losses.py](../src/model/losses.py) but is off by default — we want to isolate changes one at a time.

**Why weights.** The dataset has ~314 safe contracts and ~122 reentrant contracts, but once we go per-call-site, the ratio shifts further: a reentrant contract often has 3–8 external calls and only 1–2 of them are actually vulnerable, so the real per-hyperedge positive rate is around **1:10**, not the 1:2.57 we inherited from the paper's contract-level count. If we used the old ratio the loss would under-penalize missed positives and the model would default to "predict safe for everything."

**How the weights are computed now.** Per fold, from the actual positive/negative counts in that fold's training set:

```python
# src/evaluation/train.py :: compute_class_weights
weight[0] = 1.0
weight[1] = min(#neg / #pos, clamp=10.0)
```

The clamp at 10 prevents blow-up in folds where Slither flagged very few call sites (rare but possible). The clamp also acts as a sanity floor on precision — without it, the model would learn to predict "vulnerable" aggressively because being wrong about a safe call is almost free.

**Gradient clipping at L2 = 1.0** on every backward step. This was added after we saw a few outlier contracts with unusually large hyperedges blow up the loss late in training. Clipping is cheap insurance.

**What we are *not* doing yet.**
- No label smoothing (the labels are already noisy from detector miss — smoothing would compound it).
- No auxiliary losses (e.g. an auxiliary loss on node-level features). Could be worth trying if the current recipe plateaus.

---

## Slide 8 — Evaluation protocol (and why recall is the headline)

**Primary dataset.** `reentrancy-detection-benchmarks/benchmarks/aggregated-benchmark/` — 436 contracts, 122 labeled reentrant, 314 labeled safe.

**Cross-validation.** 3-fold, using `sklearn.model_selection.KFold(n_splits=3, shuffle=True, random_state=42)` applied *per class* (so each fold keeps the 122:314 ratio). This matches the `make_folds.py` script that ships with the benchmark repo — we replicate it verbatim for reproducibility.

**Seeds.** We run the whole pipeline under 5 random seeds `{42, 0, 1, 2, 3}` and report mean ± std. This is critical because with small folds (~40 reentrant contracts in val), seed-to-seed variance is not negligible.

**Held-out test set.** `manually-verified-reentrancy-dataset` — a second, separate repo with human-verified labels. We never touch this during training or hyperparameter tuning. It exists so we have one truly unseen evaluation at the end of the project.

**Metrics we compute.** Precision, Recall, F1, FNR (false-negative rate), FPR (false-positive rate), accuracy. Everything is reported at the **hyperedge level**, not contract level.

**Why recall / FNR is the headline.** In security, the cost of a missed vulnerability is asymmetric. If we miss a reentrancy in an audited contract and it gets deployed, the attacker drains funds and there is no undo. If we falsely flag a safe call, the auditor spends 5–10 minutes confirming it's fine. So we are willing to give up some precision to gain recall, within reason. "Within reason" means we aren't willing to predict "vulnerable" for every call — that has zero information content. The operating point we want is something like "catch ≥90% of real reentrancies while keeping precision high enough that auditors don't drown in alerts."

**Secondary metric: localization accuracy.** Eventually we want to report, for contracts we flag, whether the hyperedge we picked aligns with the human-annotated vulnerable line. This is the step-10 case study. Not implemented yet.

---

## Slide 9 — Current numbers (1 seed × 3 folds, weighted CE loss, L=2)

Data: [results/cv_summary.csv](../results/cv_summary.csv), [results/cv_detailed.csv](../results/cv_detailed.csv).

**Aggregate (mean ± std across folds):**

| Metric     | Mean   | Std    |
|------------|--------|--------|
| Precision  | 0.683  | 0.217  |
| **Recall** | **0.460** | **0.268** |
| **F1**     | **0.448** | **0.059** |
| **FNR**    | **0.540** | **0.268** |
| FPR        | 0.205  | 0.155  |
| Accuracy   | 0.541  | 0.085  |

**The real story is in the per-fold breakdown:**

| Fold | Precision | Recall | F1    | What this row means                                          |
|------|-----------|--------|-------|--------------------------------------------------------------|
| 1    | 0.91      | 0.26   | 0.40  | Model is extremely conservative. Flags very few calls. The few it flags are almost always right. But it misses 3 out of 4 real vulnerabilities. |
| 2    | 0.75      | 0.28   | 0.41  | Same regime as fold 1. Same failure mode.                    |
| 3    | 0.39      | 0.84   | 0.53  | Opposite regime. Model fires aggressively, catches most vulnerabilities, but over-flags. Less than half of its "vulnerable" calls are real. |

**How to read this.** The mean F1 of 0.45 with std 0.06 *looks* stable, but that hides the fact that precision std is 0.22 and recall std is 0.27. The model is not broken — it is producing two completely different decision boundaries on different folds. On folds 1+2 it behaves like a cautious auditor; on fold 3 it behaves like a panicked auditor. That instability is the real problem, not the average F1.

**What would the number look like if we tuned the decision threshold?** The training loop logs a diagnostic: "what threshold would maximize F1 on the train set?" On folds 1+2 it's 0.10–0.30 (i.e. the model's probabilities are poorly calibrated, and we'd need to fire much more eagerly to get F1 up). We deliberately do *not* apply this threshold to val because doing so would double-dip on the val set — we already picked the best epoch by val F1, and tuning the threshold on val too would inflate numbers artificially. Threshold tuning is deferred until regularization closes the calibration gap.

---

## Slide 10 — Current blocker: cross-fold instability

**Symptom (restated).** Same model, same hyperparameters, same loss. Fold 1: recall 0.26. Fold 3: recall 0.84. That's a 3× swing driven entirely by which contracts happened to land in validation. Something in the training pipeline is not robust.

**Hypothesis A — Label-distribution drift between folds.** Slither's reentrancy detectors have their own biases. Some contracts get cleanly flagged ("one specific call site is vulnerable"); others get over-flagged or under-flagged. If fold 3's validation set happens to contain contracts where Slither is *liberal* (flags lots of calls), training on folds 1+2 sees a stricter signal and the model learns a stricter boundary — then evaluating on fold 3's looser labels makes it look like the model is over-predicting. This is a *labeling ceiling* issue, not a model issue.

→ **Concrete check:** log per-fold positive rate on train vs val. If they differ by >2×, this hypothesis is likely dominant.

**Hypothesis B — Poor probability calibration from overfitting.** The diagnostic threshold (0.10–0.30) is a classic overfitting fingerprint. A well-calibrated model picks ~0.5 as optimal; a model where the training probabilities collapse toward the extremes picks something very different. We haven't added any regularization (dropout=0, weight_decay=0).

→ **Concrete fix to try:** dropout=0.2, weight_decay=1e-4, early-stopping patience=10 epochs.

**Hypothesis C — Feature coverage is still thin on cross-contract reentrancy.** Our `sender_controlled_target` bit is coarse: it fires on `msg.sender` as the target, but misses cases where the target is an unvalidated address passed in as a function parameter, or retrieved via an interface lookup. If fold-to-fold variation includes many cross-contract cases, the model is genuinely under-informed on those.

→ **Concrete check:** on fold 1 false negatives, eyeball the contracts. Are they single-contract reentrancies (should work) or cross-contract (feature gap)?

**Honest read.** F1 = 0.45 at std 0.06 with recall std 0.27 is **not a publishable number yet**. The architecture is sound; the training signal is not yet clean. I'd rather say that upfront than have a reviewer point it out.

---

## Slide 11 — Questions I want the room to push back on

**1. Is per-call-site labeling via Slither circular?**
We are training a model to *reproduce and ideally outperform* Slither's detectors. If we label from Slither, the upper bound of how well we can do on the training signal is basically Slither itself, minus our labeling noise. Two options:
- (a) **Accept it for the primary benchmark.** Pitch Repo 2 (the manually verified dataset) as the real test of generalization — if we beat Slither there, the circularity worry evaporates.
- (b) **Invest in manual labeling** of a small subset of Repo 1 to break the circularity and use that as the gold standard.

I lean (a) for now but want the room's take.

**2. Is the 3-fold split too coarse?**
With 122 reentrant contracts and 3 folds, each val set has ~40 reentrant contracts. That is *definitely* contributing to the cross-fold variance. Move to 5-fold? Stratify further by detector category? The paper specifies 3-fold, but we can report both.

**3. Is F1 the right thing to pick checkpoints by?**
If we're recall-first, arguably we should be selecting checkpoints by "recall at fixed precision ≥ X," not by F1. The train-optimal threshold diagnostic already hints this would change the story. Risk: picking by recall could degenerate into "always predict vulnerable."

**4. Cross-contract reentrancy is a deliberate blind spot right now.**
Our δ=3 bounded ancestor expansion is inside `G_call` of *one* contract. Calls that span contract boundaries don't get ancestor context. Do we:
- (a) Pitch this as a deliberate v1 scope restriction (single-contract reentrancy) and publish, then extend.
- (b) Build cross-contract call resolution now (significant work, pushes publication).

**5. Am I overfitting to this benchmark's idiosyncrasies?**
The Aggregated Benchmark is known to have duplicates and near-duplicates across the reentrant/safe folders. Should we run a dedup pass before CV?

---

## Slide 12 — Plan for the next two weeks

Priorities ranked by "what moves the numbers fastest" vs "what unblocks publication."

| Priority | Task                                                                         | Payoff                                              |
|----------|------------------------------------------------------------------------------|-----------------------------------------------------|
| P0       | Add dropout + weight decay + early stopping. One sweep, 5 seeds × 3 folds.   | Tests Hypothesis B directly; should close std gap. |
| P0       | Log per-fold positive rate (train vs val) + detector agreement.              | Tests Hypothesis A; quantifies the labeling ceiling. |
| P1       | Ablation A1 — replace hypergraph with clique-expansion GCN (same features).  | Answers "is the hypergraph actually pulling weight, or are the features doing all the work?" This is the ablation reviewers will demand first. |
| P1       | Ablations A2 (drop S(c)) and A3 (drop F(c) ancestors).                       | Which slice of the hyperedge carries the signal? If A2 barely hurts, S(c) is redundant — unlikely but worth confirming. |
| P2       | Held-out eval on Repo 2 using the best checkpoint from the current sweep.    | First number on truly unseen data. This is the one that decides whether the project has legs. |
| P2       | One localization case study — one true positive, one false positive.         | Qualitative evidence for the next meeting; also stress-tests the source-line-mapping code. |

Deferred: baselines (GCN/GAT/MLP/RF + Slither head-to-head) — doing them now would compare a miscalibrated HGNN against well-tuned baselines and make us look worse than we are. Do them after the regularization sweep.

---

## Slide 13 — Risks and the advisor ask

**Scope for the first paper.** I'm proposing we stay **reentrancy-only**. Arguments for: the pattern is well-defined, the benchmarks exist, the community cares. Arguments against: a multi-vulnerability pitch would be stronger for a top-tier venue. I lean reentrancy-only; want to confirm.

**Novelty claim — which of these is *the* contribution?**
- (i) The transaction-centric hyperedge definition (one exploit pattern = one hyperedge).
- (ii) Applying HGNNs to smart-contract security (application novelty).
- (iii) Per-hyperedge localization as a first-class output.

I lean (i) + (iii), with (ii) as the vehicle. But (ii) alone might not be enough for a security venue.

**Target venue.**
- Security venues (USENIX Security, CCS, NDSS, S&P): will want head-to-head tool comparisons (Slither, Mythril, Securify) and real-world case studies. Relatively low bar on ML novelty.
- ML venues (NeurIPS, ICLR, KDD): will want the ablation story, depth-sensitivity curves, comparisons to other GNN variants. Relatively low bar on security specifics.

We cannot hit both with equal depth on this timeline. Picking early lets us cut scope accordingly.

**Dynamic evidence (transaction traces).** Explicit non-goal for v1. Revisit only if reviewers insist.

---

## Appendix — How each collaborator can help in this session

- **Static analysis expertise:** is the Slither-detector-as-labeling-oracle policy (Slide 11 Q1) sound? Are there known failure modes of the reentrancy detectors I should stop ignoring?
- **GNN / ML expertise:** the calibration / fold-variance diagnosis on Slide 10. Any better regularizers than dropout+WD for small-sample graph classification?
- **Security / audit perspective:** is recall-first genuinely what auditors want, or would "precision-at-top-k" (the tool surfaces the top 5 suspicious calls per contract) be more usable in practice?
- **Fresh eyes:** does the example on Slides 3–5 land cleanly, or am I skipping a step that a non-specialist reader would trip on?
