"""
French Gemma 3 Model Architecture Wrapper.

This module defines the FrenchGemmaModel class, which wraps the base Gemma 3
transformer and appends a PyTorch-native Linear language modeling head,
supporting optional embedding noise injection (NEFTune style).
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class FrenchGemmaModel(nn.Module):
    """
    A custom PyTorch nn.Module wrapping the base Gemma 3 transformer architecture
    and adding a custom Linear LM Head mapped to the tokenizer vocabulary size.
    """

    def __init__(
        self,
        model_id: str,
        vocab_size: int,
        embedding_noise_std: float = 0.0,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.embedding_noise_std = embedding_noise_std

        # Load blank config
        self.config = AutoConfig.from_pretrained(model_id)
        self.config.dtype = torch.float32

        # Update vocabulary size based on custom tokenizer
        self.config.vocab_size = vocab_size

        # Apply overrides (like sequence length, attention sliding windows, etc.) if provided
        if config_override:
            for key, val in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, val)

        # Load the base model with blank config (no weights loaded)
        self.model = AutoModel.from_config(self.config)  # type: ignore[no-untyped-call]

        # Create LM head mapped to vocab size using PyTorch abstractions
        self.lm_head: nn.Module = nn.Linear(self.config.hidden_size, vocab_size, bias=False)

        # Tie word embeddings if configured
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def load_pretrained_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        """
        Loads weights from a local checkpoint directory or file into this model instance.
        Supports directory format (containing pytorch_model.bin, model.safetensors, or .pt)
        or direct checkpoint file paths.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

        weights_file: Optional[str] = None
        if os.path.isdir(checkpoint_path):
            for candidate in ("pytorch_model.bin", "model.safetensors", "model.pt", "checkpoint.pt"):
                cand_path = os.path.join(checkpoint_path, candidate)
                if os.path.exists(cand_path):
                    weights_file = cand_path
                    break
            if weights_file is None:
                # Find any .bin or .pt file in the directory
                for fname in os.listdir(checkpoint_path):
                    if fname.endswith((".bin", ".pt", ".safetensors")):
                        weights_file = os.path.join(checkpoint_path, fname)
                        break
            if weights_file is None:
                raise FileNotFoundError(f"No weights file found inside directory: {checkpoint_path}")
        else:
            weights_file = checkpoint_path

        logger.info(f"Loading pretrained model weights from: {weights_file}")
        state_dict: Any
        if weights_file.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(weights_file, device="cpu")
            except ImportError:
                state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        else:
            state_dict = torch.load(weights_file, map_location="cpu", weights_only=False)

        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Adapt keys if saved under different wrappers
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("module."):
                clean_k = clean_k[len("module.") :]
            if clean_k.startswith("_orig_mod."):
                clean_k = clean_k[len("_orig_mod.") :]
            cleaned_state_dict[clean_k] = v

        load_res = self.load_state_dict(cleaned_state_dict, strict=strict)
        logger.info(
            f"Loaded weights successfully. Missing keys: {len(load_res.missing_keys)}, "
            f"Unexpected: {len(load_res.unexpected_keys)}"
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens  # type: ignore[no-any-return]

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass computing LM logits and cross-entropy loss if labels are provided.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Inject Gaussian embedding noise during training if configured
        if self.training and self.embedding_noise_std > 0:
            noise = torch.randn_like(inputs_embeds) * self.embedding_noise_std
            inputs_embeds = inputs_embeds + noise

        # Pass input through the base model
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

        hidden_states = outputs.last_hidden_state

        # Compute logits via LM Head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for causal training: model predicts next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Loss function with cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
