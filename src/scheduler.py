"""
Learning Rate Restarts and Freeze Schedulers.

This module provides the FreezeManager for freezing layers over training milestones,
and constructs learning rate schedulers supporting linear warmup and cosine annealing with restarts.
"""

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class FreezeManager:
    """
    Manages dynamic parameter freezing based on a training step schedule.
    """

    def __init__(self, model: nn.Module, freeze_schedule: Dict[int, List[int]]) -> None:
        self.model = model
        # Sort schedule by step keys
        self.freeze_schedule = {int(k): list(v) for k, v in sorted(freeze_schedule.items())}
        self.last_applied_step = -1

    def step(self, current_step: int) -> None:
        """
        Checks the schedule and applies any freezing configuration target for the current step.
        """
        # Find the active configuration for the current step
        active_step = None
        for step_key in sorted(self.freeze_schedule.keys()):
            if current_step >= step_key:
                active_step = step_key
            else:
                break

        # If we have a new active schedule state that hasn't been applied yet
        if active_step is not None and active_step != self.last_applied_step:
            frozen_layers = self.freeze_schedule[active_step]
            self.apply_freeze(frozen_layers, active_step)
            self.last_applied_step = active_step

    def apply_freeze(self, frozen_layers: List[int], trigger_step: int) -> None:
        """
        Freezes target layers and unfreezes all other parameters.
        Embeddings and LM Head are kept unfrozen.
        """
        logger.info(f"Applying freeze configuration from step {trigger_step}: frozen layers = {frozen_layers}")

        frozen_params_count = 0
        active_params_count = 0

        for name, param in self.model.named_parameters():
            # Check if parameter belongs to a layer in the freeze list (e.g. "model.layers.0.")
            is_frozen = False
            for layer_idx in frozen_layers:
                # Target layers matching 'layers.<layer_idx>.' pattern
                if f"layers.{layer_idx}." in name:
                    is_frozen = True
                    break

            if is_frozen:
                param.requires_grad = False
                frozen_params_count += 1
            else:
                param.requires_grad = True
                active_params_count += 1

        logger.info(
            f"Freeze applied: {frozen_params_count} parameters frozen, " f"{active_params_count} parameters active."
        )


def get_lr_multiplier(
    step: int,
    warmup_steps: int,
    eta_min_ratio: float = 0.01,
    T_0: int = 1000,
    T_mult: int = 2,
) -> float:
    """
    Computes learning rate multiplier supporting linear warmup and cosine annealing with warm restarts.
    """
    if step < warmup_steps:
        # Linear warmup
        return float(step) / float(max(1, warmup_steps))

    # Cosine annealing with warm restarts starting after warmup
    t = step - warmup_steps

    if T_mult <= 1:
        current_period = max(1, T_0)
        cycle_step = t % current_period
    else:
        cycle_step = t
        current_period = max(1, T_0)
        while cycle_step >= current_period:
            cycle_step -= current_period
            current_period *= T_mult

    cos_out = math.cos(math.pi * float(cycle_step) / float(current_period))
    lr_ratio = eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + cos_out)
    return lr_ratio


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: Optional[int] = None,
    T_0: Optional[int] = None,
    T_mult: int = 2,
    eta_min_ratio: float = 0.01,
    total_steps: Optional[int] = None,
    warmup_ratio: float = 0.03,
    num_cycles: int = 1,
) -> LambdaLR:
    """
    Constructs a LambdaLR learning rate scheduler combining linear warmup
    and cosine annealing with warm restarts.

    If total_steps is provided and warmup_steps or T_0 are not explicitly set,
    they are automatically configured according to total_steps.
    """
    if warmup_steps is not None:
        effective_warmup_steps = warmup_steps
    elif total_steps is not None:
        effective_warmup_steps = max(1, int(total_steps * warmup_ratio))
    else:
        effective_warmup_steps = 100

    if T_0 is not None:
        effective_t0 = T_0
    elif total_steps is not None:
        anneal_steps = max(1, total_steps - effective_warmup_steps)
        if num_cycles <= 1:
            effective_t0 = anneal_steps
        else:
            if T_mult > 1:
                sum_factors = sum(T_mult**i for i in range(num_cycles))
                effective_t0 = max(1, int(anneal_steps / sum_factors))
            else:
                effective_t0 = max(1, int(anneal_steps / num_cycles))
    else:
        effective_t0 = 1000

    logger.info(
        f"Configured LR Scheduler: total_steps={total_steps}, "
        f"warmup_steps={effective_warmup_steps}, T_0={effective_t0}, "
        f"T_mult={T_mult}, num_cycles={num_cycles}, eta_min_ratio={eta_min_ratio}"
    )

    def lr_lambda(step: int) -> float:
        return get_lr_multiplier(
            step=step,
            warmup_steps=effective_warmup_steps,
            eta_min_ratio=eta_min_ratio,
            T_0=effective_t0,
            T_mult=T_mult,
        )


    return LambdaLR(optimizer, lr_lambda)

