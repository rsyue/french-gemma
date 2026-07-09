import torch

from src.config import TrainingConfig
from src.dataset import PackedTextDataset, get_dataloader, load_french_dataset, train_custom_tokenizer
from src.model import FrenchGemmaModel
from src.scheduler import FreezeManager, get_cosine_warmup_scheduler
from src.trainer import Pretrainer


def main() -> None:
    # 1. Load configuration optimized for Macbook Pro (configs/mlx_config.yaml)
    config = TrainingConfig.from_yaml("configs/mlx_config.yaml")
    print(f"Loaded config. Device target: {config.device}")

    # 2. Load dataset (falls back to a mock corpus if Wikipedia offline)
    texts = load_french_dataset(dataset_path=config.dataset_path, dataset_name=config.dataset_name, split="train[:100]")
    print(f"Loaded {len(texts)} articles/paragraphs.")

    # 3. Train a custom tokenizer on the French data
    print("Training custom ByteLevelBPETokenizer...")
    tokenizer = train_custom_tokenizer(texts, vocab_size=10000, save_dir="./tokenizer_checkpoint")

    # 4. Create packed sequences dataset and DataLoader
    dataset = PackedTextDataset(texts=texts, tokenizer=tokenizer, max_seq_len=config.max_sequence_length, stride=50)
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
    )
    print(f"Dataset packed into {len(dataset)} sequences of length {config.max_sequence_length}.")

    # 5. Initialize FrenchGemmaModel from blank configuration
    model = FrenchGemmaModel(
        model_id=config.model_id, vocab_size=len(tokenizer), embedding_noise_std=config.embedding_noise_std
    ).to(config.device)

    # 6. Configure optimizer & schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps=config.warmup_steps, T_0=1000)
    freeze_manager = FreezeManager(model, config.freeze_schedule)

    # 7. Initialize Pretrainer loop
    trainer = Pretrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        freeze_manager=freeze_manager,
        device=config.device,
        amp_enabled=config.amp_enabled,
        amp_dtype=config.amp_dtype,
        output_dir=config.output_dir,
        tb_log_dir=config.tb_log_dir,
        log_interval=10,
        eval_interval=50,
        save_interval=100,
    )

    # 8. Start training!
    print("Starting pretraining loop...")
    for i in range(5):
        print(f"==== Epoch {i+1} ====")
        trainer.train_epoch(epoch=i, global_step=(len(dataloader) * (i + 1)))
    print("Pretraining completed successfully!")


if __name__ == "__main__":
    main()
