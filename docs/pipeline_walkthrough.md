# HGNN Pipeline Walkthrough — Step-by-Step with Real Data

This document traces the entire pipeline from raw Solidity source code to HGNN prediction, using **two real contracts** and the **actual output of our code**. Every intermediate value shown here was produced by running our implementation.

Reference: `Hypergraph_Analysis.tex`, Sections 3–5.

---

## Table of Contents

1. [Overview: What the Pipeline Does](#1-overview)
2. [Example Contract: `Withdraw`](#2-example-contract)
3. [Step 1: AST, CFG, Call Graph Extraction](#3-step-1)
4. [Step 2: Data Dependency Graph G_dep](#4-step-2)
5. [Step 3: Node Set Construction (V_f, V_s, V_c)](#5-step-3)
6. [Step 4: Node Feature Matrix X](#6-step-4)
7. [Step 5: Hyperedge Construction + Incidence Matrix H](#7-step-5)
8. [Step 6: HGNN Forward Pass](#8-step-6)
9. [Training: Loss and Optimization](#9-training)
10. [Multi-Hyperedge Example: PERSONAL_BANK](#10-multi-hyperedge)
11. [Summary of Data Shapes](#11-summary)

---

## 1. Overview

The pipeline detects **reentrancy vulnerabilities** in Solidity smart contracts. For each external call site in a contract, it builds a **hyperedge** that captures the call's context (which functions are in the call chain, which state variables are read/written around the call), then classifies that hyperedge as vulnerable or safe using a **Hypergraph Neural Network (HGNN)**.

```
Solidity source (.sol)
    │
    ▼
Step 1: Slither ──► AST, CFG, G_call, call_sites, state_vars, functions
    │
    ▼
Step 2: CFG analysis ──► G_dep (bipartite: call_site → state_var)
    │
    ▼
Step 3: Naming ──► V_f, V_s, V_c, V, node_index
    │
    ▼
Step 4: Feature encoding ──► X ∈ R^{|V| × 34}
    │
    ▼
Step 5: BFS + G_dep lookup ──► E (hyperedge list), H_inc ∈ {0,1}^{|V| × |E|}
    │
    ▼
Step 6: HGNN forward pass ──► ŷ ∈ R^{|E| × 2}  (per-hyperedge softmax)
    │
    ▼
Loss: CrossEntropyLoss(ŷ, y_true) ──► backprop ──► Adam update
```

---

## 2. Example Contract

**`Withdraw` — the classic reentrancy example** (spec Section 4.7):

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Withdraw {
    mapping(address => uint256) public balance;        // line 5

    function deposit() public payable {                // line 7
        balance[msg.sender] += msg.value;              // line 8
    }

    function withdraw(uint256 amount) public {         // line 11
        require(balance[msg.sender] >= amount);        // line 12 — reads balance BEFORE call
        (bool success, ) = msg.sender.call{value: amount}("");  // line 13 — EXTERNAL CALL
        require(success);                              // line 14
        balance[msg.sender] -= amount;                 // line 15 — writes balance AFTER call
    }
}
```

**Why is this vulnerable?** The state update (`balance -= amount`) happens *after* the external call (`msg.sender.call`). An attacker's fallback function can re-enter `withdraw()` before the balance is decremented, draining the contract.

---

## 3. Step 1: AST, CFG, Call Graph Extraction

**Code**: `src/extraction/ast_cfg.py` → `extract_all(sol_path)`

**What it does**: Uses Slither to parse the Solidity file and extract:
- Function metadata (name, visibility, mutability)
- State variable metadata (name, type, storage slot)
- External call sites (function, line number, opcode, value transfer)
- Call graph G_call (which functions call which)
- Per-function CFG (control flow graph nodes with read/write tracking)

### Extracted Functions

| Function | Visibility | Mutability | Is Constructor |
|----------|-----------|------------|----------------|
| `deposit` | public | payable | False |
| `withdraw` | public | nonpayable | False |

### Extracted State Variables

| Variable | Type | Storage Slot |
|----------|------|-------------|
| `balance` | `mapping(address => uint256)` | 0 |

### Extracted External Call Sites

| Function | Line | Opcode | Has Value Transfer |
|----------|------|--------|-------------------|
| `withdraw` | 13 | `call` | True |

Only one external call site: the `msg.sender.call{value: amount}("")` at line 13 inside `withdraw()`. It uses the low-level `call` opcode and transfers value (ETH).

### Call Graph G_call

```
Nodes: [deposit, withdraw]
Edges: []  (neither function calls the other internally)
```

G_call is a `nx.DiGraph` where an edge `f_i → f_j` means `f_i` internally calls `f_j`. In this contract, `deposit` and `withdraw` are independent — no internal calls between them.

### Control Flow Graph (CFG)

The CFG is extracted per function. Each node tracks which state variables are read/written at that point, and whether an external call occurs.

**`deposit()` CFG:**

| Line | Node Type | Expression | Vars Read | Vars Written |
|------|-----------|-----------|-----------|-------------|
| 7 | ENTRYPOINT | | | |
| 8 | EXPRESSION | `balance[msg.sender] += msg.value` | `[balance]` | `[balance]` |

**`withdraw()` CFG:**

| Line | Node Type | Expression | Vars Read | Vars Written | External Call? |
|------|-----------|-----------|-----------|-------------|----------------|
| 11 | ENTRYPOINT | | | | |
| 12 | EXPRESSION | `require(balance[msg.sender] >= amount)` | `[balance]` | | |
| 13 | VARIABLE | | | | |
| 13 | EXPRESSION | `(success,None) = msg.sender.call{value: amount}()` | | | **YES** |
| 14 | EXPRESSION | `require(success)` | | | |
| 15 | EXPRESSION | `balance[msg.sender] -= amount` | `[balance]` | `[balance]` | |

**Key observation**: `balance` is read at line 12 (before the external call at line 13) and written at line 15 (after the call). This is exactly the read-before / write-after pattern that signals reentrancy.

---

## 4. Step 2: Data Dependency Graph G_dep

**Code**: `src/extraction/gdep.py` → `build_gdep(cfg, call_sites, state_vars)`

**What it does**: For each external call site `c`, looks at the CFG to find which state variables are **read before** or **written after** the call within the same function. Creates a bipartite directed graph with edges `(call_site_node → state_var_node)`.

**Algorithm for each call site `c` at line L in function `f`:**
1. Walk through the CFG nodes of function `f` in order
2. Find the CFG node at line L that has `has_external_call = True`
3. Any state variable read in CFG nodes **before** L → add edge `(c, var)`
4. Any state variable written in CFG nodes **after** L → add edge `(c, var)`

### G_dep for Withdraw

```
Nodes: [call:withdraw:13, var:balance]
Edges:
    call:withdraw:13 --> var:balance
```

**Why this edge exists**: Looking at `withdraw()`'s CFG:
- Line 12 reads `balance` → **before** the call at line 13 ✓
- Line 15 writes `balance` → **after** the call at line 13 ✓

So `balance` is data-dependent on the call site at line 13.

---

## 5. Step 3: Node Set Construction

**Code**: `src/hypergraph/nodeset.py` → `build_node_sets(functions, state_vars, call_sites)`

**What it does**: Creates three **disjoint** node sets and assigns each node a unique string identifier and integer index.

### Node Naming Convention

| Node Type | ID Format | Example |
|-----------|-----------|---------|
| Function node (V_f) | `func:<name>` | `func:withdraw` |
| State variable node (V_s) | `var:<name>` | `var:balance` |
| Call site node (V_c) | `call:<function>:<line>` | `call:withdraw:13` |

### Node Sets for Withdraw

```
V_f = [func:deposit, func:withdraw]                — 2 function nodes
V_s = [var:balance]                                 — 1 state variable node
V_c = [call:withdraw:13]                            — 1 call site node
V   = [func:deposit, func:withdraw, var:balance, call:withdraw:13]  — 4 total
```

### Node Index (maps node ID → integer position in V)

| Node | Index |
|------|-------|
| `func:deposit` | 0 |
| `func:withdraw` | 1 |
| `var:balance` | 2 |
| `call:withdraw:13` | 3 |

### Invariants (enforced by assertion)

1. **Disjoint**: `V_f ∩ V_s = ∅`, `V_f ∩ V_c = ∅`, `V_s ∩ V_c = ∅`
2. **Complete**: `|V| = |V_f| + |V_s| + |V_c|` → `4 = 2 + 1 + 1` ✓
3. **Ordered**: V is always ordered as `V_f ++ V_s ++ V_c`

---

## 6. Step 4: Node Feature Matrix X

**Code**: `src/hypergraph/features.py` → `build_feature_matrix(...)`

**What it does**: Each node gets a feature vector of dimension **d = 34**. The features are **type-specific** — function nodes use one region of the vector, state variable nodes use another, and call site nodes use a third. Unused regions are zero-padded.

The dimension grew from 26 → 34 when the **reentrancy-specific features**
(Section 4.3 extended) were added on V_s and V_c. These flags/counts carry
the defining reentrancy pattern — gas-forwarding call to an attacker-
controlled target, followed by an unguarded state write — directly into the
node features the HGNN sees.

### Feature Layout (d = 34)

```
Index:  0  1  2  3  4  5  6  7  8   9 10 11 12 13 14 15 16   17   18 19 20   21 22   23 24 25 26   27   28 29 30   31 32 33
        ├───── func features (9) ──┤  ├────── type category (8) ──────┤  slot  ├access┤  ├flags┤  ├── call opcode ─┤  val  ├ reentrancy flags + log-counts ┤
        ├─── V_f (9) ──────────────┤  ├──────────────── V_s (14) ─────────────────────────────┤  ├──────────────── V_c (11) ──────────────────────────────┤
```

#### Function Node Features [indices 0–8]

| Index | Feature | Encoding |
|-------|---------|----------|
| 0–3 | Visibility | One-hot: `[public, private, internal, external]` |
| 4–7 | Mutability | One-hot: `[pure, view, payable, nonpayable]` |
| 8 | Is Constructor | Binary: 0 or 1 |

#### State Variable Features [indices 9–22]

| Index | Feature | Encoding |
|-------|---------|----------|
| 9–16 | Type Category | One-hot: `[uint, address, bool, bytes, mapping, array, struct, other]` |
| 17 | Normalized Slot | `slot / max_slot` (continuous, 0 to 1) |
| 18–20 | Access Pattern | One-hot: `[read_only, write_only, read_write]` |
| 21 | Written After Call | Binary: var is written after some external call in the same function |
| 22 | Read Before Call | Binary: var is read before some external call in the same function |

**Type classification** maps Solidity types to categories:
- `mapping(address => uint256)` → `"mapping"` (index 13)
- `uint256`, `int128` → `"uint"` (index 9)
- `address` → `"address"` (index 10)
- `LogFile` (capitalized, not primitive) → `"struct"` (index 15)

**Access patterns** are computed from the CFG by scanning all functions:
- `read_only`: the variable is only read, never written
- `write_only`: the variable is only written, never read
- `read_write`: the variable is both read and written (like `balance`)

#### Call Site Features [indices 23–33]

| Index | Feature | Encoding |
|-------|---------|----------|
| 23–26 | Call Opcode | One-hot: `[call, delegatecall, staticcall, other]` |
| 27 | Value Transfer | Binary: 1 if ETH is sent with the call |
| 28 | Gas Forwarded | Binary: `.call`/`.delegatecall`/`.staticcall` (forwards all remaining gas — exploitable). `.send`/`.transfer` forward only 2300 gas and cannot re-enter. |
| 29 | Sender-Controlled Target | Binary: call destination is `msg.sender` or a function parameter — attacker can steer it |
| 30 | Guarded by Modifier | Binary: enclosing function carries a `nonReentrant`-style modifier |
| 31 | log1p(writes_after_call_count) | Continuous: how many state vars are written after this call in the same function |
| 32 | log1p(reads_after_call_count) | Continuous: how many are read after |
| 33 | log1p(reads_before_call_count) | Continuous: how many are read before |

### Actual Feature Vectors for Withdraw

```
X shape: (4, 34)

[0] func:deposit        nonzero: [(0, 1.0), (6, 1.0)]
    → index 0 = visibility:public, index 6 = mutability:payable

[1] func:withdraw       nonzero: [(0, 1.0), (7, 1.0)]
    → index 0 = visibility:public, index 7 = mutability:nonpayable

[2] var:balance          nonzero: [(13, 1.0), (20, 1.0), (21, 1.0), (22, 1.0)]
    → index 13 = type:mapping, index 20 = access:read_write,
      index 21 = written_after_call, index 22 = read_before_call

[3] call:withdraw:13     nonzero: [(23, 1.0), (27, 1.0), (28, 1.0), (29, 1.0),
                                    (31, log1p(1)), (33, log1p(1))]
    → index 23 = opcode:call, index 27 = value_transfer,
      index 28 = gas_forwarded (.call forwards all gas),
      index 29 = sender_controlled_target (msg.sender.call),
      index 31 = log1p(writes_after_call) > 0 (balance written after),
      index 33 = log1p(reads_before_call) > 0 (balance read before)
```

**Interpretation**: Each vector encodes exactly what the spec calls for, and
the reentrancy pattern is now directly visible in the feature matrix:
- `func:deposit` is public+payable
- `func:withdraw` is public+nonpayable
- `var:balance` is a mapping that is both read and written, **and** the
  written-after-call / read-before-call bits flag the exact reentrancy pattern
- `call:withdraw:13` uses the `call` opcode, sends ETH, forwards all gas, has
  an attacker-controlled target (`msg.sender`), is not guarded, and has state
  writes after it — every dimension of the reentrancy recipe

---

## 7. Step 5: Hyperedge Construction + Incidence Matrix H

**Code**: `src/hypergraph/hyperedges.py` → `build_hyperedges(...)`

**What it does**: For each call site `c ∈ V_c`, constructs a hyperedge `e_c` that captures the full context of the call:

```
e_c = {c} ∪ F(c) ∪ S(c)
```

where:
- **`{c}`**: the call site node itself
- **`F(c)`**: the function containing the call + its ancestors in G_call (bounded to δ=3 hops)
- **`S(c)`**: state variables that are data-dependent on the call (from G_dep)

### F(c) — Call Chain (Bounded Ancestor Expansion)

```
F(c) = {func:f} ∪ BoundedAncestors(f, G_call, δ=3)
```

**BoundedAncestors** does a BFS traversal **backwards** (following predecessors) in G_call, stopping at depth δ=3. This captures which functions can transitively reach the function containing the call site.

For `call:withdraw:13`:
- The call site is inside function `withdraw`
- `BoundedAncestors(withdraw, G_call, δ=3) = {}` (no function calls withdraw)
- `F(c) = {func:withdraw}`

### S(c) — Data-Dependent State Variables

```
S(c) = {s ∈ V_s | (c, s) ∈ G_dep}
```

Look up the successors of `c` in G_dep that are in V_s.

For `call:withdraw:13`:
- G_dep has edge `call:withdraw:13 → var:balance`
- `S(c) = {var:balance}`

### Constructed Hyperedge

```
e_0 = {c} ∪ F(c) ∪ S(c)
    = {call:withdraw:13} ∪ {func:withdraw} ∪ {var:balance}
    = {call:withdraw:13, func:withdraw, var:balance}

|e_0| = 3
```

This hyperedge captures: "The external call at line 13 happens inside `withdraw()`, and the state variable `balance` is read/written around it." This is precisely the information needed to detect reentrancy.

### Incidence Matrix H_inc

The incidence matrix H_inc is a binary matrix of shape `(|V|, |E|)`. Entry `H_inc[i, j] = 1` if node `V[i]` is a member of hyperedge `E[j]`.

```
|E| = 1 (one hyperedge, since there is one call site)
H_inc shape: (4, 1)

node                            e_0
─────────────────────────────────────
func:deposit                     0
func:withdraw                    1    ← member of e_0
var:balance                      1    ← member of e_0
call:withdraw:13                 1    ← member of e_0
```

**Key invariants** (enforced by assertions):
1. `|E| == |V_c|` — exactly one hyperedge per call site
2. `H_inc` is binary (only 0s and 1s)
3. Every call site appears in its own hyperedge: `H_inc[node_index[c], j] == 1`

---

## 8. Step 6: HGNN Forward Pass

**Code**: `src/model/hgnn.py` → `HGNN.forward(X, H_inc, E, node_index)`

**What it does**: Takes the node features X and incidence matrix H_inc, performs L layers of spectral message passing on the hypergraph, then mean-pools node embeddings per hyperedge and classifies each hyperedge as vulnerable or safe.

### Step 6a: Degree Matrices

**Node degree** D_v — how many hyperedges each node belongs to:

```
D_v(i,i) = Σ_e H_inc[i, e]

func:deposit      → degree 0  (not in any hyperedge)
func:withdraw     → degree 1  (in e_0)
var:balance       → degree 1  (in e_0)
call:withdraw:13  → degree 1  (in e_0)
```

**Hyperedge degree** D_e — how many nodes each hyperedge contains:

```
D_e(e,e) = Σ_v H_inc[v, e]

e_0 → degree 3  (contains 3 nodes)
```

### Step 6b: Normalized Incidence Matrix

```
H̃ = D_v^{-1/2} · H · D_e^{-1/2}
```

This normalizes the incidence matrix so that message passing is properly scaled. Isolated nodes (degree 0, like `func:deposit`) get 0 — they don't participate in message passing.

```
H̃ shape: (4, 1)

func:deposit       [0.0000]    ← isolated, no contribution
func:withdraw      [0.5774]    ← 1/√(1·3) = 1/√3
var:balance        [0.5774]
call:withdraw:13   [0.5774]
```

The value `0.5774 = 1/√3` comes from: node degree = 1, hyperedge degree = 3, so `1/√1 · 1 · 1/√3 = 1/√3`.

### Step 6c: The Theta Matrix (Node-to-Node Connectivity)

```
Θ = H̃ · H̃ᵀ    (shape: |V| × |V|)
```

This is the key matrix for message passing. Entry `Θ[i, j]` measures how strongly connected nodes i and j are **through shared hyperedge membership**.

```
Non-zero entries:
  Θ[withdraw, withdraw]     = 0.3333    ← self-connection via e_0
  Θ[withdraw, balance]      = 0.3333    ← connected via e_0
  Θ[withdraw, call:13]      = 0.3333    ← connected via e_0
  Θ[balance, withdraw]      = 0.3333
  Θ[balance, balance]       = 0.3333
  Θ[balance, call:13]       = 0.3333
  Θ[call:13, withdraw]      = 0.3333
  Θ[call:13, balance]       = 0.3333
  Θ[call:13, call:13]       = 0.3333
```

All three members of `e_0` are equally connected (weight 1/3 each). `func:deposit` has all zeros — it's isolated and receives no messages.

**Two-step interpretation** (spec Section 5.4):
1. `H̃ᵀ · X`: aggregate node features **into** each hyperedge (weighted by membership)
2. `H̃ · (...)`: broadcast hyperedge representations **back** to member nodes

The combined `Θ · X` effectively says: "each node's new representation is a weighted average of the features of all nodes it shares a hyperedge with."

### Step 6d: Layer-wise Message Passing (L = 2 layers)

```
Input: X^(0) = InputProjection(X)     — project from d=34 to hidden_dim=16

For each layer l = 0, 1:
    X_new = ReLU(Θ · X^(l) · W^(l))  — message passing + linear transform + activation
    X^(l+1) = X^(l) + X_new           — residual connection
    X^(l+1) = LayerNorm(X^(l+1))      — normalize

Output: Z = X^(2)                      — final node embeddings, shape (4, 16)
```

**Residual connection** ensures that the original features are preserved — each layer adds information rather than replacing it. This prevents the vanishing gradient problem in deeper networks.

**LayerNorm** normalizes across the feature dimension to stabilize training.

**W^(l)** is a learnable `Linear(16, 16, bias=False)` matrix per layer.

### Step 6e: Hyperedge Embedding (Mean Pooling)

For each hyperedge e_j, compute its embedding by **averaging** the final node embeddings of its member nodes:

```
z_{e_0} = mean(Z[func:withdraw], Z[var:balance], Z[call:withdraw:13])
```

This produces a single vector of dimension 16 that represents the hyperedge.

**Why mean pooling?** (spec Section 5.5) It treats all member nodes equally. Attention-based pooling could weight them differently, but the paper explicitly specifies mean pooling.

### Step 6f: Classifier (Softmax over 2 classes)

```
logits = W_c · z_e + b_c        — Linear(16, 2)
ŷ = softmax(logits)             — [p(safe), p(vulnerable)]
```

**Output for Withdraw (untrained model, seed=42):**

```
logits:   [0.1755, 0.6683]
softmax:  [0.3792, 0.6208]
predicted class: 1 (vulnerable)
```

The output shape is always `(|E|, 2) = (|V_c|, 2)` — one probability distribution per external call site.

### Model Architecture Summary

```
InputProjection: Linear(34 → 16, bias=False)     — 544 params
Layer 0:         Linear(16 → 16, bias=False)      — 256 params
LayerNorm 0:     LayerNorm(16)                     — 32 params
Layer 1:         Linear(16 → 16, bias=False)      — 256 params
LayerNorm 1:     LayerNorm(16)                     — 32 params
Classifier:      Linear(16 → 2)                    — 34 params
────────────────────────────────────────────────────────────────
Total:                                              1,154 params
```

---

## 9. Training: Loss and Optimization

### Label Assignment (Section 6.2 revised — per call site)

Labels are assigned **per hyperedge** (not per contract) because each hyperedge
corresponds to a single external call site, and within a reentrant contract
usually only one or two call sites are actually vulnerable:

- **Safe contract** (directory `safe/`): every hyperedge gets label **0**.
- **Reentrant contract** (directory `reentrant/`): Slither's reentrancy
  detectors (`ReentrancyEth`, `ReentrancyReadBeforeWritten`, `ReentrancyNoGas`,
  `ReentrancyBenign`, `ReentrancyEvent`) flag the specific `(function, line)`
  pairs that exhibit the vulnerability. Only those call sites get label **1**;
  the other call sites in the same contract get **0**.
- **Fallback**: if the contract is known-reentrant but Slither flags nothing
  (detector misses it), all its hyperedges fall back to label **1** so the
  positive is not lost.

For Withdraw (a vulnerable contract with one call site that Slither flags):
`y_true = [1]`.

Implementation: `src/extraction/labels.py` → `compute_call_site_labels(...)`.

### Loss Function

**Weighted Cross-Entropy Loss** with **per-fold** class weights. Once labels
are per call site, the positive rate is much lower than the contract-level
ratio (closer to 1 : 10 than 1 : 2.57), and it varies between folds:

```
class_weights = compute_class_weights(train_data, clamp=10.0)
            # = [1.0, n_neg / max(n_pos, 1)] clamped to [1.0, 10.0]
loss = CrossEntropyLoss(logits, labels, weight=class_weights)
```

For the untrained model on Withdraw:
```
logits  = [0.1755, 0.6683]
labels  = [1]
loss    = 0.4768
```

### Optimization

- **Optimizer**: Adam with lr = 1e-3
- **Training**: per-contract forward pass → loss → backward → step
- **CV**: 3-fold stratified cross-validation, 5 random seeds [42, 0, 1, 2, 3]
- **Reproducibility**: `torch.manual_seed(seed)` + `np.random.seed(seed)` at each fold

---

## 10. Multi-Hyperedge Example: PERSONAL_BANK

The `Withdraw` contract has only one call site, producing one hyperedge. Here's a more complex example: **PERSONAL_BANK**, a real-world vulnerable contract from the dataset with **3 external call sites**.

### Extracted Data

**Functions** (6):

| Function | Visibility | Mutability |
|----------|-----------|------------|
| SetMinSum | public | nonpayable |
| SetLogFile | public | nonpayable |
| Initialized | public | nonpayable |
| Deposit | public | payable |
| Collect | public | payable |
| fallback | public | payable |

**State Variables** (4):

| Variable | Type | Category | Slot |
|----------|------|----------|------|
| balances | `mapping(address => uint256)` | mapping | 0 |
| MinSum | `uint256` | uint | 1 |
| Log | `LogFile` | struct | 2 |
| intitalized | `bool` | bool | 3 |

**External Call Sites** (3):

| ID | Opcode | Value Transfer |
|----|--------|---------------|
| `call:Deposit:38` | AddMessage | No |
| `call:Collect:47` | call | Yes |
| `call:Collect:50` | AddMessage | No |

### Call Graph G_call

```
Nodes: [Collect, Deposit, Initialized, SetLogFile, SetMinSum, fallback]
Edges: [fallback → Deposit]
```

The `fallback()` function internally calls `Deposit()`.

### G_dep Edges

```
call:Collect:47 --> var:balances     (balances read before / written after call)
call:Collect:47 --> var:MinSum       (MinSum read before call)
call:Collect:50 --> var:balances
call:Collect:50 --> var:MinSum
call:Deposit:38 --> var:balances     (balances written around Deposit's call)
```

### Node Sets

```
|V_f| = 6    (6 functions)
|V_s| = 4    (4 state variables)
|V_c| = 3    (3 external call sites)
|V|   = 13
```

### Hyperedges (3)

**e_0 for `call:Deposit:38`:**
```
F(c) = {func:Deposit} ∪ BoundedAncestors(Deposit, G_call, δ=3)
     = {func:Deposit} ∪ {fallback}     ← fallback calls Deposit
     = {func:Deposit, func:fallback}

S(c) = {var:balances}                   ← from G_dep

e_0 = {call:Deposit:38, func:Deposit, func:fallback, var:balances}
|e_0| = 4
```

Note how **ancestor expansion** pulls in `func:fallback` because `fallback → Deposit` in G_call.

**e_1 for `call:Collect:47`:**
```
F(c) = {func:Collect} ∪ BoundedAncestors(Collect, G_call, δ=3)
     = {func:Collect} ∪ {}              ← no function calls Collect
     = {func:Collect}

S(c) = {var:balances, var:MinSum}       ← from G_dep

e_1 = {call:Collect:47, func:Collect, var:balances, var:MinSum}
|e_1| = 4
```

**e_2 for `call:Collect:50`:**
```
e_2 = {call:Collect:50, func:Collect, var:balances, var:MinSum}
|e_2| = 4
```

### Incidence Matrix H_inc (13 × 3)

```
node                            e_0  e_1  e_2
──────────────────────────────────────────────
func:SetMinSum                   0    0    0
func:SetLogFile                  0    0    0
func:Initialized                 0    0    0
func:Deposit                     1    0    0    ← e_0
func:Collect                     0    1    1    ← e_1, e_2
func:fallback                    1    0    0    ← e_0 (ancestor of Deposit)
var:balances                     1    1    1    ← all three hyperedges
var:MinSum                       0    1    1    ← e_1, e_2
var:Log                          0    0    0
var:intitalized                  0    0    0
call:Deposit:38                  1    0    0    ← e_0
call:Collect:47                  0    1    0    ← e_1
call:Collect:50                  0    0    1    ← e_2
```

**Observations:**
- `var:balances` is in ALL 3 hyperedges — it's the most connected state variable
- `func:Collect` is shared by e_1 and e_2 (both call sites are in the same function)
- `func:fallback` appears in e_0 due to ancestor expansion (it calls `Deposit`)
- Several nodes (SetMinSum, SetLogFile, Initialized, Log, intitalized) are isolated — they have no edges in the incidence matrix and receive no messages during HGNN propagation

### HGNN Output (3 predictions)

```
e_0 (call:Deposit:38):  softmax=[0.5353, 0.4647]  → predicted SAFE
e_1 (call:Collect:47):  softmax=[0.6314, 0.3686]  → predicted SAFE
e_2 (call:Collect:50):  softmax=[0.5336, 0.4664]  → predicted SAFE
```

These are from an **untrained** model — predictions are near random. After training, the model should learn that `call:Collect:47` (the `msg.sender.call{value: ...}` with balance read before and written after) is the actual reentrancy vulnerability.

---

## 11. Summary of Data Shapes

| Symbol | Code Name | Shape | Description |
|--------|-----------|-------|-------------|
| V | `V` | `list[str]`, length `\|V\|` | All node identifiers, ordered V_f ++ V_s ++ V_c |
| V_f | `V_f` | `list[str]`, length `\|V_f\|` | Function node identifiers |
| V_s | `V_s` | `list[str]`, length `\|V_s\|` | State variable node identifiers |
| V_c | `V_c` | `list[str]`, length `\|V_c\|` | Call site node identifiers |
| X | `X` | `np.ndarray (|V|, 34)` | Node feature matrix |
| G_call | `G_call` | `nx.DiGraph` | Call graph over function names |
| G_dep | `G_dep` | `nx.DiGraph` | Bipartite: call_site → state_var |
| E | `E` | `list[set[str]]`, length `\|E\|` = `\|V_c\|` | Hyperedge sets |
| **H** | `H_inc` | `np.ndarray (|V|, |E|)`, binary | Incidence matrix |
| H̃ | `H_tilde` | `torch.Tensor (|V|, |E|)` | Normalized: D_v^{-1/2} H D_e^{-1/2} |
| Θ | `HHt` | `torch.Tensor (|V|, |V|)` | H̃ · H̃ᵀ — node-to-node via hyperedges |
| Z | `Z` | `torch.Tensor (|V|, hidden_dim)` | Final node embeddings after L layers |
| z_e | `z_e` | `torch.Tensor (hidden_dim,)` | Hyperedge embedding (mean pool) |
| ŷ | `y_pred` | `torch.Tensor (|E|, 2)` | Softmax output [p(safe), p(vuln)] |

### For Withdraw: `|V|=4, |E|=1, d=34`
### For PERSONAL_BANK: `|V|=13, |E|=3, d=34`
