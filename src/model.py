from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class FrenchGemmaModel(nn.Module):
    """
    A custom PyTorch nn.Module wrapping the base Gemma 3 transformer architecture
    and adding a custom Linear LM Head mapped to the tokenizer vocabulary size.
    """
    def __init__(self, model_id: str, vocab_size: int, config_override: dict = None):
        super().__init__()
        
        # Load blank config
        self.config = AutoConfig.from_pretrained(model_id)
        
        # Update vocabulary size based on custom tokenizer
        self.config.vocab_size = vocab_size
        
        # Apply overrides (like sequence length, attention sliding windows, etc.) if provided
        if config_override:
            for key, val in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, val)
                    
        # Load the base model with blank config (no weights loaded)
        self.model = AutoModel.from_config(self.config)
        
        # Create LM head mapped to vocab size using PyTorch abstractions
        self.lm_head = nn.Linear(self.config.hidden_size, vocab_size, bias=False)
        
        # Tie word embeddings if configured
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
            
    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass computing LM logits and cross-entropy loss if labels are provided.
        """
        # Pass input through the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
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
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
