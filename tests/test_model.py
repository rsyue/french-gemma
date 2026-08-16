"""
Unit Tests for Model Architecture and Freeze Schedules.

This module contains unit tests validating base model wraps, LM Head projections,
embedding noise injection, layer freezing scheduler steps, and cosine annealing schedules.
"""

import os

import pytest
import torch

from src.model import FrenchGemmaModel
from src.scheduler import FreezeManager, get_cosine_warmup_scheduler


def test_model_initialization():
    vocab_size = 300
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=vocab_size)
    assert model.config.vocab_size == vocab_size
    assert model.lm_head.out_features == vocab_size

    # Assert weight tying
    if model.config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight


def test_model_forward():
    vocab_size = 200
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=vocab_size)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (2, 16))
    labels = torch.randint(0, vocab_size, (2, 16))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    assert outputs.loss is not None
    assert outputs.logits.shape == (2, 16, vocab_size)
    assert outputs.loss.item() > 0


def test_freeze_scheduler():
    vocab_size = 200
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=vocab_size)

    # Layer 0 and 1 frozen at step 0; none frozen at step 5
    freeze_schedule = {0: [0, 1], 5: []}

    manager = FreezeManager(model, freeze_schedule)

    # Apply step 0
    manager.step(0)
    # named_parameters belonging to layer 0/1 should be frozen (requires_grad = False)
    for name, param in model.named_parameters():
        if "layers.0." in name or "layers.1." in name:
            assert not param.requires_grad
        elif "layers.5." in name:
            assert param.requires_grad

    # Apply step 5
    manager.step(5)
    # All parameters should now be active
    for name, param in model.named_parameters():
        assert param.requires_grad


def test_lr_scheduler():
    vocab_size = 100
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps=10, T_0=50, T_mult=2)

    # Initial lr should be near 0 at step 0 (due to linear warmup starting at 0)
    assert optimizer.param_groups[0]["lr"] == 1e-3 * 0.0

    # Step scheduler to step 1
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0

    # Step scheduler past warmup (step 11)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    # Should be decaying cosine
    assert optimizer.param_groups[0]["lr"] < 1e-3


def test_lr_scheduler_total_steps_auto_calc():
    vocab_size = 100
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = get_cosine_warmup_scheduler(
        optimizer,
        total_steps=1000,
        warmup_ratio=0.03,
        num_cycles=1,
        eta_min_ratio=0.01,
    )

    # Step 0: Warmup start = 0
    assert optimizer.param_groups[0]["lr"] == 0.0

    # Step to end of warmup (step 30)
    for _ in range(30):
        optimizer.step()
        scheduler.step()

    # At step 30 (end of warmup), LR should reach peak (1e-3)
    assert abs(optimizer.param_groups[0]["lr"] - 1.0e-3) < 1e-6

    # Step to 999th step (969th step of annealing cycle)
    for _ in range(969):
        optimizer.step()
        scheduler.step()

    # At step 999 (end of annealing cycle), LR should reach minimum eta_min_ratio (1e-3 * 0.01 = 1e-5)
    assert abs(optimizer.param_groups[0]["lr"] - 1.0e-5) < 1e-6




def test_embedding_noise():
    vocab_size = 100
    # Initialize model with non-zero embedding noise std
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=vocab_size, embedding_noise_std=0.5)

    input_ids = torch.randint(0, vocab_size, (2, 8))

    # 1. In eval mode, outputs should be identical across calls (no noise injected)
    model.eval()
    with torch.no_grad():
        out1 = model(input_ids)
        out2 = model(input_ids)
    assert torch.allclose(out1.logits, out2.logits, atol=1e-5)

    # 2. In train mode, outputs should differ due to noise injection
    model.train()
    with torch.no_grad():
        out3 = model(input_ids)
        out4 = model(input_ids)
    assert not torch.allclose(out3.logits, out4.logits, atol=1e-4)

    # 3. If noise std is 0, train mode should produce identical outputs (no noise)
    model_no_noise = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=vocab_size, embedding_noise_std=0.0)
    model_no_noise.train()
    with torch.no_grad():
        out5 = model_no_noise(input_ids)
        out6 = model_no_noise(input_ids)
    assert torch.allclose(out5.logits, out6.logits, atol=1e-5)


def test_ensure_tokenizer_vocab_alignment():
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=200)
    assert model.lm_head.out_features == 200
    assert model.model.embed_tokens.num_embeddings == 200

    # Align with new tokenizer length
    model.ensure_tokenizer_vocab_alignment(350)
    assert model.lm_head.out_features == 350
    assert model.model.embed_tokens.num_embeddings == 350
    assert model.config.vocab_size == 350
    if model.config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight


def test_compare_and_load_automodel_raises_on_vocab_mismatch(tmp_path):
    # Create source model with vocab_size 500
    src_model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=500)
    ckpt_dir = str(tmp_path / "ckpt_500")
    os.makedirs(ckpt_dir, exist_ok=True)
    weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    torch.save(src_model.state_dict(), weights_path)

    # Destination model with vocab_size 250 (e.g. SFT tokenizer length)
    dst_model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=250)

    # Must raise ValueError due to vocabulary size mismatch between checkpoint and tokenizer
    with pytest.raises(ValueError, match="Vocabulary / embedding dimension mismatch"):
        dst_model.load_pretrained_checkpoint(ckpt_dir)


def test_compare_and_load_automodel_raises_on_layer_shape_mismatch():
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=250)

    # Corrupt a layer's tensor shape in checkpoint state dict
    fake_state_dict = model.state_dict()
    fake_state_dict["model.norm.weight"] = torch.randn(10)  # Incorrect shape

    with pytest.raises(ValueError, match="Architecture layer size / hidden dimension mismatch"):
        model.compare_and_load_automodel(fake_state_dict)


def test_load_matching_checkpoint_succeeds(tmp_path):
    src_model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=250)
    ckpt_file = str(tmp_path / "matching_model.pt")
    torch.save(src_model.state_dict(), ckpt_file)

    dst_model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=250)
    dst_model.load_pretrained_checkpoint(ckpt_file)

    for p1, p2 in zip(src_model.parameters(), dst_model.parameters()):
        assert torch.allclose(p1, p2)


def test_load_hf_automodel_checkpoint_format(tmp_path):
    # Simulating a HuggingFace checkpoint where state_dict has keys directly for AutoModel (no model. prefix)
    src_model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=300)
    hf_automodel_state = src_model.model.state_dict()

    ckpt_file = str(tmp_path / "hf_model.pt")
    torch.save(hf_automodel_state, ckpt_file)

    dst_model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=300)
    dst_model.load_pretrained_checkpoint(ckpt_file)

    for k in hf_automodel_state.keys():
        assert torch.allclose(dst_model.model.state_dict()[k], hf_automodel_state[k])

