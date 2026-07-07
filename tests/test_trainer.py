import os
import tempfile

import torch

from src.dataset import PackedTextDataset, get_dataloader, train_custom_tokenizer
from src.model import FrenchGemmaModel
from src.scheduler import FreezeManager, get_cosine_warmup_scheduler
from src.trainer import Pretrainer, generate_text


def test_trainer_integration():
    mock_texts = [
        "Ceci est un texte de test en français pour vérifier l'intégration.",
        "Le processus de pré-entraînement doit se dérouler sans erreur.",
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup tokenizer
        tok_dir = os.path.join(tmpdir, "tokenizer")
        tokenizer = train_custom_tokenizer(mock_texts, vocab_size=150, save_dir=tok_dir)
        vocab_size = len(tokenizer)
        
        # 2. Setup dataset and loader
        dataset = PackedTextDataset(mock_texts, tokenizer, max_seq_len=8, stride=2)
        dl = get_dataloader(dataset, batch_size=2, num_workers=0, pin_memory=False)
        
        # 3. Setup model
        model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=vocab_size)
        
        # 4. Setup optimizers & schedulers
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps=1, T_0=10)
        freeze_manager = FreezeManager(model, {0: [0]})
        
        # 5. Setup Trainer
        chk_dir = os.path.join(tmpdir, "checkpoints")
        run_dir = os.path.join(tmpdir, "runs")
        
        trainer = Pretrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=dl,
            val_dataloader=dl,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            freeze_manager=freeze_manager,
            device="cpu",
            amp_enabled=False,
            amp_dtype="float32",
            grad_clip_norm=1.0,
            grad_accum_steps=1,
            log_interval=1,
            eval_interval=2,
            save_interval=2,
            output_dir=chk_dir,
            tb_log_dir=run_dir
        )
        
        # 6. Train for 2 steps
        global_step = trainer.train_epoch(epoch=0, global_step=0)
        assert global_step > 0
        
        # 7. Check check-pointing
        assert len(os.listdir(chk_dir)) > 0
        
        # 8. Test generate
        generated = generate_text(model, tokenizer, "Test", max_new_tokens=5, device="cpu")
        assert len(generated) > 0

def test_best_checkpoint_rotation():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal structures for constructor
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
            def state_dict(self):
                return {"weights": torch.ones(10)}
                
        class MockOptimizer:
            def state_dict(self):
                return {}
                
        trainer = Pretrainer(
            model=MockModel(),
            tokenizer=None, # not used in save_best_checkpoint
            train_dataloader=None,
            val_dataloader=None,
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            tb_log_dir=os.path.join(tmpdir, "runs")
        )
        
        # Save 4 checkpoints with decreasing perplexity (lower is better)
        trainer.save_best_checkpoint(global_step=1, perplexity=100.0)
        trainer.save_best_checkpoint(global_step=2, perplexity=80.0)
        trainer.save_best_checkpoint(global_step=3, perplexity=60.0)
        
        # We should have 3 checkpoints saved
        chkpts = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts) == 3
        
        # Save a 4th one that is better (ppl=40)
        trainer.save_best_checkpoint(global_step=4, perplexity=40.0)
        
        # Should still have only 3 checkpoints (the one with ppl=100.0 should be deleted)
        chkpts_after = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts_after) == 3
        assert not any("ppl-100.00" in name for name in chkpts_after)
        assert any("ppl-40.00" in name for name in chkpts_after)
