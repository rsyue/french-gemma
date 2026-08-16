# French Gemma 3 Pretraining Library

A modular, extensible PyTorch library to train a blank `Gemma 3` model (default architecture: `google/gemma-3-270m-it`) from scratch on unsupervised French data using HuggingFace `transformers` configurations, custom byte-level tokenizers, and PyTorch training optimization loops.

This project implements architectures and practices from the **Gemma 3 Paper**: [Gemma 3: Open Models Built with Google's Gemini Technology (Google DeepMind, 2024)](https://arxiv.org/abs/2503.19786).

---

## Key Features

1.  **Modular & Extensible Training Framework (`train/`)**: Core training features are decoupled into an extensible strategy/trainer hierarchy supporting dependency injection, protocol contracts, and custom training algorithms (e.g. pretraining, future RLHF/DPO extensions).
2.  **Decoder-Only Causal Training**: Configures and instantiates a blank Gemma 3 model, adding a PyTorch-native `nn.Linear` LM Head.
3.  **Custom Tokenization**: Trains a local byte-level BPE tokenizer (`ByteLevelBPETokenizer`) from scratch to correctly handle French contractions, elisions (e.g. `l'`, `d'`), and accents. Uses **right padding** (`padding_side = "right"`) optimized for causal decoder training.
4.  **Sliding Window Packing & Sentinel Synchronization**: Packs token sequences separated by `<bos>` and `<eos>` tokens into fixed `max_sequence_length` inputs using atomic disk-backed binary cache packing with `.ready` sentinel synchronization across distributed ranks.
5.  **Gaussian Embedding Noise**: Introduces adjustable Gaussian noise (via `embedding_noise_std`) directly into word embeddings during training (NEFTune style) to boost generalization and robustness. Noise is bypassed automatically during evaluation.
6.  **Un-frozen Pretraining by Default**: Runs pretraining without layer freezing by default for max model capacity (optional layer freezing schedules can still be configured).
7.  **Robust Distributed Pretraining (DDP)**: Defers `dist.init_process_group` until rank 0 finishes data preparation, utilizing filesystem sentinel polling and configurable `dist_timeout_seconds` to eliminate NCCL communication timeouts.
8.  **Advanced Training Loops**: Full support for mixed-precision (AMP) training, gradient accumulation, gradient clipping, AdamW optimizer, and Cosine Annealing with Warm Restarts and Warmup.
9.  **Log & Checkpoint Management**: Reports progress to TensorBoard and retains only the **three best checkpoints** based on validation perplexity.
10. **Proportional Multi-Dataset Mixing**: Supports configuring multi-dataset mixtures (e.g. Wikipedia 50%, OSCAR 30%, C4 20%) with percentage weights via inline YAML or standalone `--data-mix` configs with deterministic interleaving.
11. **Interactive Streaming Inference**: Terminal-based chat interface (`inference.py`) with streaming decoding and configurable sampling controls (`do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`).

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
├── train/
│   ├── __init__.py             # Package exports (BaseTrainer, TrainingFactory, PretrainStrategy)
│   ├── base.py                 # Abstract base classes, protocols, and StrategyType unions
│   ├── builder.py              # ModularTrainer and dependency injection TrainingFactory
│   ├── cli.py                  # Dynamic CLI argument parser & TrainingConfig plumbing
│   ├── pretrain.py             # CLI entrypoint executable via python -m train.pretrain
│   └── strategies/
│       ├── base.py             # AbstractTrainingStrategy base definition
│       └── pretrain.py         # Causal Language Model pretraining strategy
├── scripts/
│   ├── run_ddp.sh              # Multi-GPU launcher helper
│   └── training_example.py     # Standalone simple training example script
├── src/
│   ├── config.py               # YAML configuration parser, data mix schemas, and dataclass
│   ├── dataset.py              # Tokenizer, dataset packing, multi-mix loading, and DataLoaders
│   ├── model.py                # FrenchGemmaModel wrapper with LM head & noise injection
│   ├── scheduler.py            # Cosine learning rate restarts and layer freezing
│   └── trainer.py              # Core pretraining engine loop and checkpointing
├── inference.py                # Interactive CLI streaming chat interface
├── tests/                      # Comprehensive unit and integration test suite
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

## Pretraining Execution (`python -m train.pretrain`)

The primary way to run pretraining is via the `train.pretrain` module. All configuration flags can be plumbed directly through the command line to override YAML defaults:

```bash
# Run pretraining using macOS MPS config with CLI parameter overrides
source .venv/bin/activate && python -m train.pretrain --config configs/mlx_config.yaml --model google/gemma-3-270m-it --batch-size 4 --learning-rate 2e-4
```

### Full Dataset Training & Custom Dataset Splits
By default, pretraining runs on the full unsupervised dataset (`num_examples: "all"`). To train on a specific subset for rapid testing or debugging, pass `--num-examples 100` (or `--num_examples 100`). The standalone `scripts/training_example.py` script defaults to 100 examples for quick smoke testing.

```bash
# Run pretraining on a subset of 100 examples for rapid debugging
source .venv/bin/activate && python -m train.pretrain --config configs/nvidia_config.yaml --num-examples 100 --vocab-size 35000
```

### Multi-GPU Pretraining Run (DDP via `torchrun`)
To train French Gemma 3 across multiple GPUs using Distributed Data Parallel (DDP):
```bash
source .venv/bin/activate && ./scripts/run_ddp.sh --config configs/nvidia_config.yaml --gpus 2
```
Or launch directly using `torchrun`:
```bash
source .venv/bin/activate && torchrun --nproc_per_node=2 -m train.pretrain --config configs/nvidia_config.yaml --num-examples all
```

> [!NOTE]
> During multi-GPU runs, Rank 0 handles dataset tokenization and binary packing while worker ranks poll for the `.ready` sentinel file. `torch.distributed.init_process_group` is deferred until data preparation completes, eliminating NCCL heartbeat/watchdog timeouts. The distributed initialization timeout can be adjusted via `dist_timeout_seconds` in your YAML config.

### Multi-Dataset Mixing (`data_mix`)
You can configure a proportional dataset mix directly in your training YAML configuration or pass a standalone mix file using `--data-mix`:

```yaml
# configs/my_custom_config.yaml
model_id: "google/gemma-3-270m-it"
num_examples: 50000
data_mix:
  - dataset_path: "wikimedia/wikipedia"
    dataset_name: "20231101.fr"
    percentage: 50.0
  - dataset_path: "oscar-corpus/OSCAR-2201"
    dataset_name: "fr"
    percentage: 30.0
  - dataset_path: "c4"
    dataset_name: "fr"
    percentage: 20.0
```

Or override dynamically via CLI:
```bash
source .venv/bin/activate && python -m train.pretrain --config configs/nvidia_config.yaml --data-mix configs/mix_config.yaml --num-examples 10000
```

---

## Inference & Interactive Chat (`inference.py`)

Run an interactive terminal chat session with text streaming decoding using a pretrained or fine-tuned Gemma 3 model checkpoint:

```bash
# Launch interactive streaming chat with default generation parameters
source .venv/bin/activate && python inference.py --model google/gemma-3-270m-it
```

### Configurable Generation Sampling Parameters
All generation hyperparameters can be customized from the command line:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `google/gemma-3-270m-it` | Model identifier or local checkpoint path |
| `--max-len` | `int` | `2048` | Context window length for generation |
| `--dtype` | `str` | `bfloat16` | PyTorch dtype (`bfloat16`, `float16`, `float32`) |
| `--do-sample` / `--no-sample` | `bool` | `True` | Whether to use sampling instead of greedy decoding |
| `--temperature` | `float` | `0.7` | Temperature for logit modulation |
| `--top-p` | `float` | `0.95` | Nucleus sampling probability threshold |
| `--top-k` | `int` | `65` | Top-k tokens considered during sampling |
| `--repetition-penalty` | `float` | `1.5` | Penalty factor for repeating tokens |

Example with custom sampling parameters:
```bash
source .venv/bin/activate && python inference.py --model ./checkpoints/best_model --temperature 0.8 --top-p 0.90 --top-k 50 --repetition-penalty 1.3
```

---

## Contributing Invitation

We welcome contributions to the French Gemma 3 training toolkit! Whether you are looking to optimize CUDA/ROCm compilation, add new training strategies to `train/strategies/`, or enhance tokenization, we would love your help.

### How to Contribute
1.  **Fork** the repository and clone your fork.
2.  Set up the environment using `uv venv` and `uv pip install -e ".[dev]"`.
3.  Implement your changes, following the local guidelines defined in [AGENTS.md](AGENTS.md).
4.  Write matching tests in the `tests/` directory.
5.  Verify that all tests, type checks, and lint checks pass cleanly (`pytest tests/`, `ruff check .`, `mypy .`).
6.  Open a **Pull Request** detailing your enhancements.

Happy coding!
