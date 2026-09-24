"""Training losses for MedLatent latent-interface supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diagnosis_cross_entropy(
    first_token_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute CE including the first answer token.

    ``first_token_logits`` are the host logits after the final question prompt.
    ``target_logits`` are logits while teacher-forcing the answer sequence.
    ``target_labels`` are answer token ids with ``ignore_index`` for padding.
    """

    if first_token_logits.ndim != 2:
        raise ValueError("first_token_logits must be [batch, vocab]")
    if target_logits.ndim != 3 or target_labels.ndim != 2:
        raise ValueError("target_logits must be [batch, target_len, vocab] and target_labels [batch, target_len]")
    first_labels = target_labels[:, :1]
    shifted_logits = target_logits[:, :-1, :]
    shifted_labels = target_labels[:, 1:]
    logits = torch.cat([first_token_logits.unsqueeze(1), shifted_logits], dim=1)
    labels = torch.cat([first_labels, shifted_labels], dim=1)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=ignore_index, reduction="none",
    ).reshape(labels.shape)
    valid = labels != ignore_index
    per_example = token_loss.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    if reduction == "none":
        return per_example
    return per_example.mean() if reduction == "mean" else per_example.sum()
