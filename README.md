# French Gemma 3 Pretraining Library

A modular, extensible PyTorch library to train a blank `Gemma 3` model (default architecture: `google/gemma-3-270m-it`) from scratch on unsupervised French data using HuggingFace `transformers` configurations, custom byte-level tokenizers, and PyTorch training optimization loops.

This project implements architectures and practices from the **Gemma 3 Paper**: [Gemma 3: Open Models Built with Google's Gemini Technology (Google DeepMind, 2024)](https://arxiv.org/abs/2412.00000).

---

## Key Features

1.  **Decoder-Only Causal Training**: Configures and instantiates a blank Gemma 3 model, adding a PyTorch-native `nn.Linear` LM Head.
2.  **Custom Tokenization**: Trains a local byte-level BPE tokenizer (`ByteLevelBPETokenizer`) from scratch to correctly handle French contractions, elisions (e.g. `l'`, `d'`), and accents. Uses **right padding** (`padding_side = "right"`) optimized for causal decoder training.
3.  **Sliding Window Packing**: Packs token sequences separated by `<bos>` and `<eos>` tokens into fixed `max_sequence_length` inputs. Employs a sliding window with a 50-token overlap stride to prevent cutting important context.
4.  **Gaussian Embedding Noise**: Introduces adjustable Gaussian noise (via `embedding_noise_std`) directly into word embeddings during training (NEFTune style) to boost generalization and robustness. Noise is bypassed automatically during evaluation.
5.  **Freeze Schedules**: Dynamically freezes/unfreezes layers at configurable step thresholds to stabilize early pretraining.
6.  **Advanced Training Loops**: Full support for mixed-precision (AMP) training, gradient accumulation, gradient clipping, AdamW optimizer, and Cosine Annealing with Warm Restarts and Warmup.
7.  **Log & Checkpoint Management**: Reports progress to TensorBoard and retains only the **three best checkpoints** based on validation perplexity.

---

## Directory Structure

```text
french_gemma/
├── configs/
│   ├── mlx_config.yaml         # macOS (MPS, bfloat16, pin_memory disabled)
│   ├── amd_config.yaml         # AMD ROCm (CUDA, bfloat16, compiled model)
│   └── nvidia_config.yaml      # Nvidia Jetson Orin Nano (CUDA, float16)
├── hooks/
│   └── README.md               # Developer environment commands and rules
├── src/
│   ├── config.py               # YAML configuration parser
│   ├── dataset.py              # Tokenizer, packed datasets, and Dataloaders
│   ├── model.py                # FrenchGemmaModel wrapper with LM head & noise injection
│   ├── scheduler.py            # Cosine learning rate restarts and layer freezing
│   └── trainer.py              # Train/evaluation loop, AMP, and checkpointing
├── tests/
│   ├── test_dataset.py         # Tokenizer and dataset packing unit tests
│   ├── test_model.py           # Model outputs, freeze manager, and noise unit tests
│   └── test_trainer.py         # Mock pretraining loop integration test
├── pyproject.toml              # Project dependencies and tool configurations
└── README.md                   # This documentation file
```

---

## Setup & Environment Setup (using `uv`)

This project uses the fast Python packaging tool `uv` to manage the virtual environment and dependencies.

### 1. Initialize Virtual Environment
Create the virtual environment using `uv`:
```bash
# Create a virtual environment in .venv/
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Install Dependencies in Dev / Editable Mode
For users looking to contribute or run training, install the package in **editable mode** along with all development and testing dependencies:
```bash
# Install dependencies and local package in editable mode with dev extra
uv pip install -e ".[dev]"
```

### 3. Run Tests
Verify the environment and implementation by running the test suite:
```bash
source .venv/bin/activate && pytest tests/
```

### 4. Run Linter
Confirm that the code conforms to style and naming requirements:
```bash
source .venv/bin/activate && ruff check .
```

### 5. Run Type Checker
Verify that there are no static type errors:
```bash
source .venv/bin/activate && mypy .
```

---

## Sample Training Run (Mac / Apple Silicon)

Below is a complete, self-contained Python script illustrating how to run a mock pretraining loop on macOS using the MPS GPU device.

Create a training script (e.g. `run_training.py`) with the following content:

```python
import torch
from src.config import TrainingConfig
from src.dataset import load_french_dataset, train_custom_tokenizer, PackedTextDataset, get_dataloader
from src.model import FrenchGemmaModel
from src.scheduler import FreezeManager, get_cosine_warmup_scheduler
from src.trainer import Pretrainer

def main():
    # 1. Load configuration optimized for Macbook Pro (configs/mlx_config.yaml)
    config = TrainingConfig.from_yaml("configs/mlx_config.yaml")
    print(f"Loaded config. Device target: {config.device}")

    # 2. Load dataset (falls back to a mock corpus if Wikipedia offline)
    texts = load_french_dataset(
        dataset_path=config.dataset_path,
        dataset_name=config.dataset_name,
        split="train[:100]"
    )
    print(f"Loaded {len(texts)} articles/paragraphs.")

    # 3. Train a custom tokenizer on the French data
    print("Training custom ByteLevelBPETokenizer...")
    tokenizer = train_custom_tokenizer(texts, vocab_size=1000, save_dir="./tokenizer_checkpoint")
    
    # 4. Create packed sequences dataset and DataLoader
    dataset = PackedTextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=config.max_sequence_length,
        stride=50
    )
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory
    )
    print(f"Dataset packed into {len(dataset)} sequences of length {config.max_sequence_length}.")

    # 5. Initialize FrenchGemmaModel from blank configuration
    model = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=len(tokenizer),
        embedding_noise_std=config.embedding_noise_std
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
        save_interval=100
    )

    # 8. Start training!
    print("Starting pretraining loop...")
    trainer.train_epoch(epoch=0, global_step=0)
    print("Pretraining completed successfully!")

if __name__ == "__main__":
    main()
```

Run the training script in single-process mode inside the virtual environment:
```bash
source .venv/bin/activate && python scripts/run_training.py --config configs/mlx_config.yaml
```

### Multi-GPU Pretraining Run (DDP via `torchrun`)
To train French Gemma 3 using multiple GPUs, use the launcher helper:
```bash
source .venv/bin/activate && ./scripts/run_ddp.sh --config configs/nvidia_config.yaml --gpus 2
```
Or launch directly using `torchrun`:
```bash
source .venv/bin/activate && torchrun --nproc_per_node=2 scripts/run_training.py --config configs/nvidia_config.yaml
```

---

## Contributing Invitation

We welcome contributions to the French Gemma 3 training toolkit! Whether you are looking to optimize CUDA ROCm compilation, experiment with larger model variants, or enhance the dataset tokenization logic, we would love your help.

### How to Contribute
1.  **Fork** the repository and clone your fork.
2.  Set up the environment using `uv venv` and `uv pip install -e ".[dev]"`.
3.  Implement your changes, following the local guidelines defined in [AGENTS.md](AGENTS.md).
4.  Write matching tests in the `tests/` directory.
5.  Verify that all tests and lint checks pass cleanly.
6.  Open a **Pull Request** detailing your enhancements.

Feel free to open an issue to discuss design options or ask questions. Happy coding!
