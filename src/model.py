"""
French Gemma 3 Model Architecture Wrapper.

This module defines the FrenchGemmaModel class, which wraps the base Gemma 3
transformer and appends a PyTorch-native Linear language modeling head,
supporting optional embedding noise injection (NEFTune style).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast
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

        self.model_id = model_id
        self.embedding_noise_std = embedding_noise_std
        self.vocab_size = vocab_size

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

        self.model: Any = AutoModel.from_config(self.config)  # type: ignore[no-untyped-call]
        self.lm_head: Any = nn.Linear(self.config.hidden_size, vocab_size, bias=False)

        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def ensure_tokenizer_vocab_alignment(
        self, tokenizer: Union[int, PreTrainedTokenizerFast, Any]
    ) -> None:
        """
        Ensures that the length of the tokenizer corresponds to the size of
        the final linear layer (self.lm_head) and embed_tokens.
        """
        target_vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else int(tokenizer)
        current_lm_out = getattr(self.lm_head, "out_features", None)

        if current_lm_out != target_vocab_size or self.config.vocab_size != target_vocab_size:
            logger.info(
                f"Aligning model vocabulary with tokenizer length: "
                f"lm_head {current_lm_out} -> {target_vocab_size}"
            )
            self.config.vocab_size = target_vocab_size
            self.vocab_size = target_vocab_size

            # Resize embed_tokens
            old_embed = self.model.embed_tokens.weight.data
            hidden_size = self.config.hidden_size
            device = old_embed.device
            dtype = old_embed.dtype
            new_embed = nn.Embedding(target_vocab_size, hidden_size, device=device, dtype=dtype)
            n_copy = min(old_embed.shape[0], target_vocab_size)
            new_embed.weight.data[:n_copy] = old_embed[:n_copy]
            self.model.embed_tokens = new_embed

            # Recreate lm_head
            self.lm_head = nn.Linear(
                hidden_size, target_vocab_size, bias=False, device=device, dtype=dtype
            )
            if getattr(self.config, "tie_word_embeddings", True):
                self.lm_head.weight = self.model.embed_tokens.weight

    def compare_and_load_automodel(
        self,
        checkpoint_state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Extracts and compares base model weights individually as an AutoModel.
        Strictly verifies that hidden dimensions, module layers, and tensor shapes match
        the expected architecture, raising ValueError if any layer mismatch is detected.
        """
        automodel_target = self.model.state_dict()
        automodel_keys = set(automodel_target.keys())

        automodel_sub_dict: Dict[str, torch.Tensor] = {}
        matched_keys: List[str] = []

        for raw_k, v in checkpoint_state_dict.items():
            k = raw_k[len("model.") :] if raw_k.startswith("model.") else raw_k

            if k in automodel_keys:
                target_shape = automodel_target[k].shape
                if v.shape != target_shape:
                    if k == "embed_tokens.weight":
                        raise ValueError(
                            f"Vocabulary / embedding dimension mismatch for AutoModel 'embed_tokens.weight': "
                            f"checkpoint has shape {v.shape} (vocab_size={v.shape[0]}, hidden_dim={v.shape[1]}), "
                            f"but model expects {target_shape} (vocab_size={target_shape[0]}, "
                            f"hidden_dim={target_shape[1]}). Ensure that post-training uses the exact "
                            f"tokenizer and model configuration from pretraining."
                        )
                    raise ValueError(
                        f"Architecture layer size / hidden dimension mismatch for AutoModel module '{k}': "
                        f"checkpoint has shape {v.shape}, but model expects {target_shape} "
                        f"(expected hidden_dim={self.config.hidden_size})."
                    )
                automodel_sub_dict[k] = v
                matched_keys.append(k)

        missing_keys = [k for k in automodel_keys if k not in automodel_sub_dict]
        if missing_keys and strict:
            raise ValueError(
                f"Checkpoint is missing {len(missing_keys)} required AutoModel module parameters "
                f"(first missing: '{missing_keys[0]}'). "
                f"Ensure the checkpoint was saved from a compatible base model ({self.model_id})."
            )

        load_res = self.model.load_state_dict(automodel_sub_dict, strict=strict)
        logger.info(
            f"AutoModel comparison verified successfully: {len(matched_keys)} module parameters matched."
        )
        return {
            "matched_keys": matched_keys,
            "missing_keys": load_res.missing_keys,
        }

    def load_pretrained_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> None:
        """
        Loads an existing model checkpoint saved locally from pretraining.
        
        Performs strict post-training compatibility checks:
        1. Compares the base HuggingFace model individually as an AutoModel, ensuring
           hidden dimensions and number of modules match exactly.
        2. Ensures the checkpoint's final linear layer (lm_head) and embeddings dimension
           strictly correspond to the tokenizer vocabulary size.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

        weights_file: Optional[str] = None
        if os.path.isdir(checkpoint_path):
            priority_candidates = ("pytorch_model.bin", "model.safetensors", "model.pt", "checkpoint.pt")
            for candidate in priority_candidates:
                cand_path = os.path.join(checkpoint_path, candidate)
                if os.path.exists(cand_path):
                    weights_file = cand_path
                    break
            if weights_file is None:
                # Find any valid model weight file in directory, ignoring training state metadata
                ignored_names = {"training_state.pt", "optimizer.pt", "scheduler.pt"}
                for fname in sorted(os.listdir(checkpoint_path)):
                    if fname.endswith((".bin", ".pt", ".safetensors")) and fname not in ignored_names:
                        weights_file = os.path.join(checkpoint_path, fname)
                        break
            if weights_file is None:
                raise FileNotFoundError(f"No valid weights file found inside directory: {checkpoint_path}")
        else:
            weights_file = checkpoint_path

        logger.info(f"Loading and validating pretrained model weights from: {weights_file}")
        state_dict: Any
        if weights_file.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(weights_file, device="cpu")
            except ImportError as err:
                raise ImportError(
                    "The package 'safetensors' is required to load .safetensors checkpoints. "
                    "Install it via 'uv pip install safetensors'."
                ) from err
        else:
            state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Invalid state dict loaded from {weights_file}: expected dictionary, got {type(state_dict)}"
            )

        # Adapt keys if saved under different wrappers (module., _orig_mod.)
        cleaned_state_dict: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            clean_k = k
            while clean_k.startswith(("module.", "_orig_mod.")):
                if clean_k.startswith("module."):
                    clean_k = clean_k[len("module.") :]
                elif clean_k.startswith("_orig_mod."):
                    clean_k = clean_k[len("_orig_mod.") :]
            cleaned_state_dict[clean_k] = v

        # 1. Compare and load base AutoModel individually with strict dimension checks
        self.compare_and_load_automodel(cleaned_state_dict, strict=strict)

        # 2. Check final linear layer (self.lm_head) and ensure dimensions match tokenizer vocab
        expected_vocab_size = self.config.vocab_size
        if "lm_head.weight" in cleaned_state_dict:
            ckpt_head = cleaned_state_dict["lm_head.weight"]
            if ckpt_head.shape[0] != expected_vocab_size:
                raise ValueError(
                    f"Final linear layer (lm_head) vocabulary mismatch: "
                    f"checkpoint has output dimension {ckpt_head.shape[0]}, "
                    f"but tokenizer length is {expected_vocab_size}."
                )
            if ckpt_head.shape[1] != self.config.hidden_size:
                raise ValueError(
                    f"Final linear layer (lm_head) hidden dimension mismatch: "
                    f"checkpoint has hidden_dim {ckpt_head.shape[1]}, "
                    f"but model expects {self.config.hidden_size}."
                )
            if not getattr(self.config, "tie_word_embeddings", True):
                self.lm_head.load_state_dict({"weight": ckpt_head})
            else:
                self.lm_head.weight = self.model.embed_tokens.weight
        elif getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens  # type: ignore[no-any-return]

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head  # type: ignore[no-any-return]

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

        loss: Optional[torch.FloatTensor] = None
        if labels is not None:
            # Shift logits and labels for causal training: model predicts next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Loss function with functional cross entropy; return zero loss connected to graph if no valid targets exist
            if (shift_labels != -100).any():
                loss_t = torch.nn.functional.cross_entropy(
                    shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
                )
                loss = loss_t.float()  # type: ignore[assignment]
            else:
                loss = (shift_logits.sum() * 0.0).float()

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
