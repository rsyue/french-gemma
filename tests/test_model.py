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
    freeze_schedule = {
        0: [0, 1],
        5: []
    }
    
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
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0
    
    # Step scheduler past warmup (step 11)
    for _ in range(10):
        scheduler.step()
    # Should be decaying cosine
    assert optimizer.param_groups[0]["lr"] < 1e-3
