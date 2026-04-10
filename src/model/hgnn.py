"""
Step 6 — Hypergraph Neural Network (HGNN) Model.
Spec: Section 5 (Hypergraph Neural Network).

Implements spectral convolution on hypergraphs with residual connections,
optional LayerNorm, mean pooling for hyperedge embeddings, and softmax classifier.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HGNN(nn.Module):
    """
    Hypergraph Neural Network for vulnerability classification.
    Spec: Section 5 — HGNN.

    Architecture:
        1. Compute D_v, D_e degree matrices
        2. Compute H_tilde = D_v^{-1/2} @ H_inc @ D_e^{-1/2}
        3. L layers: X = X + sigma(H_tilde @ H_tilde^T @ X @ W[l]), optional LayerNorm
        4. Mean pool node embeddings per hyperedge -> z_e
        5. Softmax classifier -> (|E|, 2)

    Args:
        in_dim: input feature dimension d
        hidden_dim: hidden dimension d' (same across all layers)
        n_layers: number of message passing layers L (default 2)
        use_layernorm: whether to apply LayerNorm after residual (default True)
        dropout: dropout rate (default 0.0)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_layernorm = use_layernorm
        self.dropout = dropout

        # Projection from input dim to hidden dim (if they differ)
        if in_dim != hidden_dim:
            self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        else:
            self.input_proj = None

        # W[l] per layer: linear transform without bias
        # Spec: W^(l) in R^{d x d}
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(n_layers)
        ])

        # Optional LayerNorm per layer
        if use_layernorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(n_layers)
            ])
        else:
            self.layer_norms = None

        # Classifier head: W_c in R^{2 x d'}, b_c in R^2
        # Spec: Section 5.6 — softmax(W_c @ z_e + b_c)
        self.classifier = nn.Linear(hidden_dim, 2)

    def _compute_H_tilde(self, H_inc: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized incidence matrix.
        Spec: Section 5.3 — H_tilde = D_v^{-1/2} @ H @ D_e^{-1/2}

        Handles isolated nodes (D_v[i,i] = 0) by setting inverse to 0.

        Args:
            H_inc: binary incidence matrix, shape (|V|, |E|)

        Returns:
            H_tilde: normalized incidence matrix, shape (|V|, |E|)
        """
        # D_v: node degree — sum over hyperedges
        # Spec: Section 5.2 — D_v(i,i) = sum_e H(i,e)
        d_v = H_inc.sum(dim=1)  # (|V|,)

        # D_e: hyperedge degree — sum over nodes
        # Spec: Section 5.2 — D_e(e,e) = sum_v H(v,e)
        d_e = H_inc.sum(dim=0)  # (|E|,)

        # Inverse square roots, handle zeros
        d_v_inv_sqrt = torch.zeros_like(d_v)
        nonzero_v = d_v > 0
        d_v_inv_sqrt[nonzero_v] = d_v[nonzero_v].pow(-0.5)

        d_e_inv_sqrt = torch.zeros_like(d_e)
        nonzero_e = d_e > 0
        d_e_inv_sqrt[nonzero_e] = d_e[nonzero_e].pow(-0.5)

        # H_tilde = D_v^{-1/2} @ H @ D_e^{-1/2}
        # Efficient: element-wise multiply with diagonal vectors
        H_tilde = d_v_inv_sqrt.unsqueeze(1) * H_inc * d_e_inv_sqrt.unsqueeze(0)

        return H_tilde

    def forward(
        self,
        X: torch.Tensor,
        H_inc: torch.Tensor,
        E: list[set[str]],
        node_index: dict[str, int],
    ) -> torch.Tensor:
        """
        Forward pass of the HGNN.
        Spec: Section 5 — Full HGNN pipeline.

        Args:
            X: node feature matrix, shape (|V|, d)
            H_inc: binary incidence matrix, shape (|V|, |E|)
            E: list of hyperedges (each a set of node identifiers)
            node_index: maps node identifier to integer index

        Returns:
            y_pred: softmax probabilities, shape (|E|, 2)
        """
        # Step 1-3: Compute normalized incidence
        H_tilde = self._compute_H_tilde(H_inc)

        # Precompute H_tilde @ H_tilde^T (shared across layers)
        # Shape: (|V|, |V|)
        HHt = H_tilde @ H_tilde.T

        # Project input to hidden dim if needed
        if self.input_proj is not None:
            X = self.input_proj(X)

        # Step 4: L layers of message passing with residual
        # Spec: Section 5.4 — X^(l+1) = X^(l) + sigma(H_tilde @ H_tilde^T @ X^(l) @ W^(l))
        for l in range(self.n_layers):
            # Message passing: H_tilde @ H_tilde^T @ X @ W[l]
            X_new = HHt @ X
            X_new = self.layers[l](X_new)
            X_new = F.relu(X_new)

            if self.dropout > 0 and self.training:
                X_new = F.dropout(X_new, p=self.dropout, training=True)

            # Residual connection
            X = X + X_new

            # Optional LayerNorm
            # Spec: Section 5.4 — LayerNorm(X^(l+1))
            if self.layer_norms is not None:
                X = self.layer_norms[l](X)

        # Step 5: Z = X^(L) — final node embeddings
        Z = X  # shape (|V|, hidden_dim)

        # Step 6: Mean pooling per hyperedge
        # Spec: Section 5.5 — z_e = (1/|e|) * sum_{v in e} Z_v
        n_edges = len(E)
        z_e_list = []
        for e in E:
            indices = []
            for node in e:
                if node in node_index:
                    indices.append(node_index[node])
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=Z.device)
                z_e = Z[idx_tensor].mean(dim=0)
            else:
                z_e = torch.zeros(self.hidden_dim, device=Z.device)
            z_e_list.append(z_e)

        # Stack: (|E|, hidden_dim)
        z_E = torch.stack(z_e_list, dim=0)

        # Step 7: Classifier
        # Spec: Section 5.6 — softmax(W_c @ z_e + b_c)
        logits = self.classifier(z_E)  # (|E|, 2)
        y_pred = F.softmax(logits, dim=1)

        return y_pred

    def forward_logits(
        self,
        X: torch.Tensor,
        H_inc: torch.Tensor,
        E: list[set[str]],
        node_index: dict[str, int],
    ) -> torch.Tensor:
        """
        Forward pass returning raw logits (for use with nn.CrossEntropyLoss).

        Args:
            Same as forward()

        Returns:
            logits: raw scores, shape (|E|, 2)
        """
        H_tilde = self._compute_H_tilde(H_inc)
        HHt = H_tilde @ H_tilde.T

        if self.input_proj is not None:
            X = self.input_proj(X)

        for l in range(self.n_layers):
            X_new = HHt @ X
            X_new = self.layers[l](X_new)
            X_new = F.relu(X_new)

            if self.dropout > 0 and self.training:
                X_new = F.dropout(X_new, p=self.dropout, training=True)

            X = X + X_new

            if self.layer_norms is not None:
                X = self.layer_norms[l](X)

        Z = X

        n_edges = len(E)
        z_e_list = []
        for e in E:
            indices = []
            for node in e:
                if node in node_index:
                    indices.append(node_index[node])
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=Z.device)
                z_e = Z[idx_tensor].mean(dim=0)
            else:
                z_e = torch.zeros(self.hidden_dim, device=Z.device)
            z_e_list.append(z_e)

        z_E = torch.stack(z_e_list, dim=0)
        logits = self.classifier(z_E)

        return logits
