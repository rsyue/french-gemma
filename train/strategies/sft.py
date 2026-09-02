"""
Supervised Fine-Tuning (SFT) Training Strategy with Prompt Masking.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from train.base import AbstractTrainingStrategy


class SFTStrategy(AbstractTrainingStrategy):
    """Supervised Fine-Tuning strategy computing cross-entropy loss on unmasked response tokens."""

    def compute_loss(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("Batch must contain 'input_ids'")

        labels = batch.get("labels")
        if labels is None:
            raise ValueError("Batch must contain 'labels' with prompt masking for SFT training")

        attention_mask = batch.get("attention_mask", None)

        kwargs: Dict[str, Any] = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = model(**kwargs)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss: torch.Tensor = outputs.loss
            return loss

        logits: Any = getattr(outputs, "logits", outputs)
        if isinstance(logits, tuple):
            logits = logits[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if not (shift_labels != -100).any():
            zero_loss: torch.Tensor = (shift_logits.sum() * 0.0).float()
            return zero_loss
        return cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
