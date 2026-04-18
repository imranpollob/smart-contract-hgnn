"""
Loss functions for HGNN vulnerability classification.

Focal loss (Lin et al., 2017) is the primary addition for Step 3: it down-
weights well-classified examples so the network keeps learning from the hard
positives instead of plateauing once easy negatives are covered. Paired with
per-class weights derived from the training label distribution (α_t), it
replaces vanilla weighted CrossEntropy when the train/val F1 gap is large
(overfit → poor calibration).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss with per-class α weighting.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where p_t is the predicted probability of the true class. At γ=0 this
    reduces to standard weighted cross-entropy.

    Args:
        gamma: focusing parameter; higher values concentrate more on hard
            examples. 2.0 is the value used in the original paper and the
            Step 3 default.
        weight: per-class α tensor of shape (C,). Typically the output of
            compute_class_weights(). If None, α_t ≡ 1 for all classes.
        reduction: "mean" | "sum" | "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.gamma = gamma
        self.reduction = reduction
        # Register as a buffer so .to(device) moves it with the module.
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw scores, shape (N, C).
            targets: int64 class indices, shape (N,).

        Returns:
            Scalar loss (or per-sample if reduction="none").
        """
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_factor = (1.0 - pt).clamp(min=0.0, max=1.0).pow(self.gamma)
        loss = -focal_factor * log_pt

        alpha_t = None
        if self.weight is not None:
            alpha_t = self.weight.to(loss.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            # Match F.cross_entropy(weight=...) semantics: weighted mean
            # (divide by sum of per-sample weights), so gamma=0 is a drop-in
            # for nn.CrossEntropyLoss(weight=...).
            if alpha_t is not None:
                return loss.sum() / alpha_t.sum().clamp(min=1e-12)
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
