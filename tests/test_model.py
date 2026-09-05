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


def test_ensure_tokenizer_vocab_alignment_preserves_dtype():
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=200)
    model.to(dtype=torch.float16)

    assert model.model.embed_tokens.weight.dtype == torch.float16
    assert model.lm_head.weight.dtype == torch.float16

    # Align with a larger vocab size
    model.ensure_tokenizer_vocab_alignment(250)

    assert model.config.vocab_size == 250
    assert model.lm_head.out_features == 250
    assert model.model.embed_tokens.weight.dtype == torch.float16
    assert model.lm_head.weight.dtype == torch.float16


def test_is_rocm_available_detection(monkeypatch):
    import src.model as model_mod

    # Simulate ROCm environment: torch.version.hip is string and cuda is available
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert model_mod.is_rocm_available() is True

    # Simulate CUDA environment: torch.version.hip is None
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert model_mod.is_rocm_available() is False

    # Simulate CPU environment: torch.version.hip is string but cuda is False
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert model_mod.is_rocm_available() is False


def test_french_gemma_model_patches_rotary_on_rocm(monkeypatch, caplog):
    import logging

    import src.model as model_mod

    # Force is_rocm_available to return True
    monkeypatch.setattr(model_mod, "is_rocm_available", lambda: True)

    with caplog.at_level(logging.INFO):
        model = model_mod.FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=100)

    assert getattr(model, "is_rocm_patched", False) is True
    assert "AMD ROCm GPU hardware detected. Patching Gemma 3 rotary embedding forward pass" in caplog.text


def test_french_gemma_model_skips_patch_when_not_rocm(monkeypatch, caplog):
    import logging

    import src.model as model_mod

    # Force is_rocm_available to return False
    monkeypatch.setattr(model_mod, "is_rocm_available", lambda: False)

    with caplog.at_level(logging.INFO):
        model = model_mod.FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=100)

    assert getattr(model, "is_rocm_patched", False) is False
    assert "AMD ROCm GPU hardware detected. Patching Gemma 3 rotary embedding forward pass" not in caplog.text


def test_patched_rotary_embedding_numerical_parity(monkeypatch):
    import src.model as model_mod

    # Unpatched model
    monkeypatch.setattr(model_mod, "is_rocm_available", lambda: False)
    unpatched_model = model_mod.FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=100)

    # Patched model
    monkeypatch.setattr(model_mod, "is_rocm_available", lambda: True)
    patched_model = model_mod.FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=100)

    x = torch.randn(2, 64, 1152)
    position_ids = torch.arange(64).unsqueeze(0).expand(2, -1)

    cos_unpatched, sin_unpatched = unpatched_model.model.rotary_emb(x, position_ids, "sliding_attention")
    cos_patched, sin_patched = patched_model.model.rotary_emb(x, position_ids, "sliding_attention")

    assert torch.allclose(cos_unpatched, cos_patched, atol=1e-6)
    assert torch.allclose(sin_unpatched, sin_patched, atol=1e-6)


def test_diagnose_non_finite_gradients_clean():
    from src.model import diagnose_non_finite_gradients

    layer = torch.nn.Linear(4, 2)
    layer.weight.grad = torch.zeros(2, 4)
    layer.bias.grad = torch.ones(2)

    msg = diagnose_non_finite_gradients(layer)
    assert "No non-finite gradients detected" in msg


def test_diagnose_non_finite_gradients_reports_offending_tensors():
    from src.model import diagnose_non_finite_gradients

    layer = torch.nn.Linear(4, 2)
    layer.weight.grad = torch.tensor([[float("nan"), 0.0, 1.0, 2.0], [0.0, float("inf"), 1.0, 2.0]])
    layer.bias.grad = torch.zeros(2)

    msg = diagnose_non_finite_gradients(layer)
    assert "'weight'" in msg
    assert "NaNs=1" in msg
    assert "Infs=1" in msg
    assert "'bias'" not in msg


def test_french_gemma_model_zero_loss_with_infinite_logits(monkeypatch):
    import src.model as model_mod

    model = model_mod.FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=64)

    # Patch lm_head to inject an infinite logit in forward output
    orig_lm_head = model.lm_head

    class InfHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = orig_lm_head

        def forward(self, x):
            res = self.head(x)
            res.data[0, 0, 0] = float("inf")
            return res

    monkeypatch.setattr(model, "lm_head", InfHead())

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, -100, -100]], dtype=torch.long)

    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    assert not torch.isnan(out.loss), "Loss must not be NaN even with infinite logits"
    assert not torch.isinf(out.loss), "Loss must not be Inf even with infinite logits"
    assert out.loss.item() == 0.0

    # Ensure backward produces finite zero gradients without error
    out.loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()




