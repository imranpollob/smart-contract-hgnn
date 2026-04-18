"""Tests for src/model/losses.py — focal loss for Step 3."""

import pytest
import torch
import torch.nn.functional as F

from src.model.losses import FocalLoss


class TestFocalLossBasics:
    def test_gamma_zero_matches_weighted_ce(self):
        """γ=0 reduces focal loss to standard weighted cross-entropy."""
        torch.manual_seed(0)
        logits = torch.randn(8, 2)
        targets = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        weight = torch.tensor([1.0, 3.0])

        focal = FocalLoss(gamma=0.0, weight=weight)
        fl = focal(logits, targets)
        ce = F.cross_entropy(logits, targets, weight=weight)
        assert torch.allclose(fl, ce, atol=1e-6)

    def test_gamma_two_down_weights_easy_examples(self):
        """At γ=2, easy (high-p_t) examples contribute far less than hard ones."""
        easy_logits = torch.tensor([[-5.0, 5.0]])  # very confident, correct
        hard_logits = torch.tensor([[-0.1, 0.1]])  # near-uniform, correct
        target = torch.tensor([1])

        focal = FocalLoss(gamma=2.0, reduction="none")
        easy = focal(easy_logits, target).item()
        hard = focal(hard_logits, target).item()
        assert hard > easy
        # Should be at least an order of magnitude: focal factor (1-p)^2 for
        # easy is tiny (p ≈ 1), for hard is near 0.25.
        assert hard / max(easy, 1e-12) > 100

    def test_weight_moves_with_model(self):
        """`weight` must be registered as buffer so .to(device) moves it."""
        w = torch.tensor([1.0, 4.0])
        loss_fn = FocalLoss(gamma=2.0, weight=w)
        # Buffer should be accessible via state_dict and module attributes.
        assert "weight" in dict(loss_fn.named_buffers())

    def test_reduction_none_preserves_shape(self):
        logits = torch.randn(5, 2)
        targets = torch.zeros(5, dtype=torch.long)
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        per_sample = loss_fn(logits, targets)
        assert per_sample.shape == (5,)

    def test_reduction_sum_equals_mean_times_n_unweighted(self):
        """Without weights, mean reduction is a plain 1/N average of losses."""
        torch.manual_seed(1)
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        mean = FocalLoss(gamma=2.0)(logits, targets)
        total = FocalLoss(gamma=2.0, reduction="sum")(logits, targets)
        assert torch.allclose(total / 10.0, mean, atol=1e-6)

    def test_weighted_mean_matches_weighted_ce_semantics(self):
        """When weight is given, mean divides by sum of per-sample alphas
        (matches F.cross_entropy). This is what makes gamma=0 a drop-in."""
        torch.manual_seed(2)
        logits = torch.randn(8, 2)
        targets = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        weight = torch.tensor([1.0, 3.0])

        mean = FocalLoss(gamma=2.0, weight=weight)(logits, targets)
        total = FocalLoss(gamma=2.0, weight=weight, reduction="sum")(logits, targets)
        alpha_sum = weight[targets].sum()
        assert torch.allclose(total / alpha_sum, mean, atol=1e-6)

    def test_gradient_flows(self):
        """Loss must be differentiable end-to-end."""
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 1, 0])
        loss = FocalLoss(gamma=2.0)(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            FocalLoss(gamma=2.0, reduction="median")


class TestFocalLossClassImbalance:
    """Focal should penalize misclassified positives more when α[1] is large."""

    def test_alpha_weighting_amplifies_positive_loss(self):
        """Per-class α scales the contribution of each sample's loss.

        Use reduction='sum' to observe the raw scaling — weighted 'mean'
        divides by sum(α_t), which cancels the scaling for single-class
        batches.
        """
        # Misclassified positive (logit favors class 0, target is 1).
        logits = torch.tensor([[3.0, -3.0]])
        target = torch.tensor([1])

        unweighted = FocalLoss(gamma=2.0, reduction="sum")(logits, target).item()
        alpha = torch.tensor([1.0, 5.0])
        weighted = FocalLoss(gamma=2.0, weight=alpha, reduction="sum")(logits, target).item()

        assert weighted == pytest.approx(5.0 * unweighted, rel=1e-6)
