"""
Unit and Integration Tests for the Pretraining Trainer.

This module contains unit and mock integration tests that verify text generation functionality,
single training epochs, validation evaluation runs, and checkpoint saving/rotation logic.
"""

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
            tb_log_dir=run_dir,
        )

        # 6. Train for 2 steps
        global_step = trainer.train_epoch(epoch=0, global_step=0)
        assert global_step > 0

        # 7. Check check-pointing
        assert len(os.listdir(chk_dir)) > 0

        # 8. Test generate
        generated = generate_text(model, tokenizer, "Test", max_new_tokens=5, device="cpu")
        assert len(generated) > 0
        trainer.close()


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
            tokenizer=None,  # not used in save_best_checkpoint
            train_dataloader=None,
            val_dataloader=None,
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            tb_log_dir=os.path.join(tmpdir, "runs"),
            max_checkpoints=2,
        )

        # Save 3 checkpoints with decreasing perplexity (lower is better)
        trainer.save_best_checkpoint(global_step=1, metric=100.0)
        trainer.save_best_checkpoint(global_step=2, metric=80.0)

        # We should have 2 checkpoints saved
        chkpts = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts) == 2

        # Save a 3rd one that is better (ppl=40)
        trainer.save_best_checkpoint(global_step=3, metric=40.0)

        # Should still have only 2 checkpoints (the one with ppl=100.0 should be deleted)
        chkpts_after = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts_after) == 2
        assert not any("ppl-100.00" in name for name in chkpts_after)
        assert any("ppl-40.00" in name for name in chkpts_after)
        trainer.close()


def test_top5_checkpoint_rotation():
    with tempfile.TemporaryDirectory() as tmpdir:
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
            tokenizer=None,
            train_dataloader=None,
            val_dataloader=None,
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            tb_log_dir=os.path.join(tmpdir, "runs"),
            max_checkpoints=5,
        )

        # Save 5 checkpoints with decreasing perplexity
        for step, ppl in enumerate([100.0, 90.0, 80.0, 70.0, 60.0], start=1):
            trainer.save_best_checkpoint(global_step=step, metric=ppl)

        chkpts = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts) == 5

        # 6th checkpoint that is worse than all 5 (ppl=120.0) -> should not be saved
        saved = trainer.save_best_checkpoint(global_step=6, metric=120.0)
        assert not saved
        chkpts = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts) == 5
        assert not any("step-6" in name for name in chkpts)

        # 7th checkpoint that is better than the worst (ppl=50.0) -> replaces step 1 (ppl=100.0)
        saved = trainer.save_best_checkpoint(global_step=7, metric=50.0)
        assert saved
        chkpts_after = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts_after) == 5
        assert not any("ppl-100.00" in name for name in chkpts_after)
        assert any("step-7" in name for name in chkpts_after)
        trainer.close()


def test_best_loss_checkpoint_rotation():
    with tempfile.TemporaryDirectory() as tmpdir:

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
            tokenizer=None,
            train_dataloader=None,
            val_dataloader=None,
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            tb_log_dir=os.path.join(tmpdir, "runs"),
            max_checkpoints=3,
        )

        # Save 4 checkpoints with decreasing training loss (lower is better)
        trainer.save_best_loss_checkpoint(global_step=1, train_loss=1.5)
        trainer.save_best_loss_checkpoint(global_step=2, train_loss=1.2)
        trainer.save_best_loss_checkpoint(global_step=3, train_loss=0.9)

        # We should have 3 checkpoints saved
        chkpts = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts) == 3

        # Save a 4th one that is better (loss=0.5)
        trainer.save_best_loss_checkpoint(global_step=4, train_loss=0.5)

        # Should still have only 3 checkpoints (the one with loss=1.5000 should be deleted)
        chkpts_after = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts_after) == 3
        assert not any("loss-1.5000" in name for name in chkpts_after)
        assert any("loss-0.5000" in name for name in chkpts_after)
        trainer.close()


def test_trainer_max_eval_batches():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.forward_calls = 0

        def state_dict(self):
            return {}

        def forward(self, input_ids, attention_mask=None, labels=None):
            self.forward_calls += 1

            class Outputs:
                loss = torch.tensor(1.0)
                logits = torch.ones(1, 1, 10)

            return Outputs()

    class MockOptimizer:
        def state_dict(self):
            return {}

    class MockTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        unk_token_id = 3

        def encode(self, text, add_special_tokens=True):
            return [1]

        def decode(self, ids, skip_special_tokens=True):
            return "Le français est"

    mock_batch = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
    val_dataloader = [mock_batch] * 5

    with tempfile.TemporaryDirectory() as tmpdir:
        model = MockModel()
        trainer = Pretrainer(
            model=model,
            tokenizer=MockTokenizer(),
            train_dataloader=None,
            val_dataloader=val_dataloader,
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            tb_log_dir=os.path.join(tmpdir, "runs"),
            max_eval_batches=2,
        )

        perplexity = trainer.evaluate(global_step=1)
        assert perplexity is not None
        # 2 eval batches + 20 generation steps in generate_text
        assert model.forward_calls == 22
        trainer.close()


def test_periodic_checkpoint_rotation():
    with tempfile.TemporaryDirectory() as tmpdir:

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def state_dict(self):
                return {"weight": torch.ones(10)}

        class MockOptimizer:
            def state_dict(self):
                return {}

        trainer = Pretrainer(
            model=MockModel(),
            tokenizer=None,
            train_dataloader=None,
            val_dataloader=None,
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            tb_log_dir=os.path.join(tmpdir, "runs"),
            max_checkpoints=2,
        )

        trainer.save_checkpoint(global_step=100)
        trainer.save_checkpoint(global_step=200)
        trainer.save_checkpoint(global_step=300)

        chkpts = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint-step-")]
        assert len(chkpts) == 2
        assert "checkpoint-step-100" not in chkpts
        assert "checkpoint-step-200" in chkpts
        assert "checkpoint-step-300" in chkpts
        trainer.close()


def test_generate_text_repetition_penalty():
    class MockTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [5]

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, input_ids, attention_mask=None, labels=None):
            local_logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 10)
            local_logits[:, -1, 5] = 10.0
            local_logits[:, -1, 6] = 8.0
            
            class Outputs:
                pass
            out = Outputs()
            out.logits = local_logits
            return out

    model = MockModel()
    tokenizer = MockTokenizer()

    # 1. No repetition penalty (repetition_penalty=1.0)
    generated_no_penalty = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="start",
        max_new_tokens=3,
        device="cpu",
        repetition_penalty=1.0,
    )
    assert generated_no_penalty == "5 5 5 5"

    # 2. With repetition penalty (repetition_penalty=1.5)
    generated_with_penalty = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="start",
        max_new_tokens=1,
        device="cpu",
        repetition_penalty=1.5,
    )
    assert generated_with_penalty == "5 6"


def test_format_step_denominator():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

    trainer = Pretrainer(
        model=MockModel(),
        tokenizer=None,
        train_dataloader=None,
        val_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        freeze_manager=None,
        device="cpu",
        max_steps=2730,
    )
    assert trainer.format_step(1) == "1/2730"
    assert trainer.format_step(50) == "50/2730"

    trainer_no_max = Pretrainer(
        model=MockModel(),
        tokenizer=None,
        train_dataloader=None,
        val_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        freeze_manager=None,
        device="cpu",
        max_steps=None,
    )
    assert trainer_no_max.format_step(1) == "1"


def test_cumulative_average_train_loss():
    """Verify average train loss is normalized per step and accumulated over total steps across epochs."""
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, input_ids, **kwargs):
            class Out:
                loss = torch.tensor(4.0, requires_grad=True)
            return Out()

    class MockOptimizer:
        def step(self):
            pass
        def zero_grad(self):
            pass
        def param_groups(self):
            return [{"lr": 1e-4}]

    class MockLoader:
        def __init__(self, num_batches):
            self.num_batches = num_batches
            self.batch_size = 2
        def __len__(self):
            return self.num_batches
        def __iter__(self):
            for _ in range(self.num_batches):
                yield {"input_ids": torch.tensor([[1, 2], [3, 4]])}

    with tempfile.TemporaryDirectory() as tmpdir:
        model = MockModel()
        opt = MockOptimizer()
        opt.param_groups = [{"lr": 1e-4}]
        dl = MockLoader(3)

        trainer = Pretrainer(
            model=model,
            tokenizer=None,
            train_dataloader=dl,
            val_dataloader=None,
            optimizer=opt,
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            grad_accum_steps=2,
            log_interval=1,
            output_dir=tmpdir,
            tb_log_dir=tmpdir,
        )

        step1 = trainer.train_epoch(epoch=0, global_step=0)
        assert trainer.total_train_steps == 2
        # Step 1 (2 batches of loss 4.0): step_loss = 4.0. Step 2 (1 batch of loss 4.0): step_loss = 4.0
        assert abs(trainer.latest_train_loss - 4.0) < 1e-5
        avg_loss = trainer.total_train_loss / trainer.total_train_steps
        assert abs(avg_loss - 4.0) < 1e-5

        # Train epoch 1: total_train_steps should accumulate to 4 without resetting
        trainer.train_epoch(epoch=1, global_step=step1)
        assert trainer.total_train_steps == 4

        trainer.close()



