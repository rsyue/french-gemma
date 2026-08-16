"""
Direct Preference Optimization (DPO) Training Strategy.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.functional import logsigmoid

from src.dpo_dataset import get_batch_logps
from train.base import AbstractTrainingStrategy


class DPOStrategy(AbstractTrainingStrategy):
    """Direct Preference Optimization strategy comparing policy model with reference model."""

    def __init__(
        self,
        ref_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> None:
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.latest_metrics: Dict[str, float] = {}

    def compute_loss(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        chosen_input_ids = batch.get("chosen_input_ids")
        chosen_labels = batch.get("chosen_labels")
        chosen_mask = batch.get("chosen_attention_mask")

        rejected_input_ids = batch.get("rejected_input_ids")
        rejected_labels = batch.get("rejected_labels")
        rejected_mask = batch.get("rejected_attention_mask")

        if chosen_input_ids is None or rejected_input_ids is None:
            raise ValueError("Batch must contain both 'chosen_input_ids' and 'rejected_input_ids'")

        if chosen_labels is None or rejected_labels is None:
            raise ValueError("Batch must contain both 'chosen_labels' and 'rejected_labels'")

        # Concatenate chosen and rejected sequences for forward pass efficiency
        batch_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        batch_attention_mask = None
        if chosen_mask is not None and rejected_mask is not None:
            batch_attention_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

        all_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        # Policy forward pass
        policy_outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        policy_logits: Any = getattr(policy_outputs, "logits", policy_outputs)
        if isinstance(policy_logits, tuple):
            policy_logits = policy_logits[0]

        bs = chosen_input_ids.shape[0]
        all_policy_logps = get_batch_logps(policy_logits, all_labels)
        policy_chosen_logps = all_policy_logps[:bs]
        policy_rejected_logps = all_policy_logps[bs:]

        # Reference forward pass
        if self.ref_model is not None:
            ref_params = list(self.ref_model.parameters())
            if ref_params and ref_params[0].device != batch_input_ids.device:
                self.ref_model.to(batch_input_ids.device)

            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=batch_input_ids, attention_mask=batch_attention_mask
                )
                ref_logits: Any = getattr(ref_outputs, "logits", ref_outputs)
                if isinstance(ref_logits, tuple):
                    ref_logits = ref_logits[0]

                all_ref_logps = get_batch_logps(ref_logits, all_labels)
                ref_chosen_logps = all_ref_logps[:bs]
                ref_rejected_logps = all_ref_logps[bs:]
        else:
            # If no reference model is provided, reference logps default to zeros
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # DPO implicit reward margin calculation
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = self.beta * (pi_logratios - ref_logratios)

        # Loss calculation with optional label smoothing
        if self.label_smoothing == 0.0:
            losses = -logsigmoid(logits)
        else:
            losses = (
                -logsigmoid(logits) * (1 - self.label_smoothing)
                - logsigmoid(-logits) * self.label_smoothing
            )

        loss = losses.mean()

        # Compute implicit rewards for metrics tracking
        chosen_rewards = (self.beta * (policy_chosen_logps - ref_chosen_logps)).detach()
        rejected_rewards = (self.beta * (policy_rejected_logps - ref_rejected_logps)).detach()
        reward_acc = (chosen_rewards > rejected_rewards).float().mean().item()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()

        self.latest_metrics = {
            "dpo_loss": loss.item(),
            "reward_accuracy": reward_acc,
            "reward_margin": reward_margin,
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }

        return loss
