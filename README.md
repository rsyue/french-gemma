# French Gemma 3 Pretraining Library

A modular, extensible PyTorch library to train a blank `Gemma 3` model (default: `google/gemma-3-270m-it` architecture) from scratch on unsupervised French data using HuggingFace `transformers` configurations, custom byte-level tokenizers, and PyTorch training optimization loops.

---

## Key Features

1.  **Decoder-Only Causal Training**: Configures and instantiates a blank Gemma 3 model, adding a PyTorch-native `nn.Linear` LM Head.
2.  **Custom Tokenization**: Trains a local byte-level BPE tokenizer (`ByteLevelBPETokenizer`) from scratch to correctly handle French contractions, elisions (e.g. `l'`, `d'`), and accents. Uses **right padding** (`padding_side = "right"`) optimized for decoder causal training.
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

## Setup & Getting Started

### 1. Initialize Virtual Environment
Set up your virtual environment and install project dependencies in editable mode:
```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Run Tests
Verify the environment and implementation by running the test suite:
```bash
source .venv/bin/activate && pytest tests/
```

### 3. Run Linter
Confirm that the code conforms to styling requirements:
```bash
source .venv/bin/activate && ruff check .
```

---

## Usage Configuration

Pretraining settings are driven by YAML configurations under `configs/`. You can load configs in your training scripts as follows:

```python
from src.config import TrainingConfig
from src.model import FrenchGemmaModel

# Load Macbook Pro config
config = TrainingConfig.from_yaml("configs/mlx_config.yaml")

# Initialize model
model = FrenchGemmaModel(
    model_id=config.model_id,
    vocab_size=len(tokenizer),
    embedding_noise_std=config.embedding_noise_std
)
```
