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
9.  **Combined Evaluation & Top-5 Checkpoint Management**: Reports progress at regular log intervals and runs evaluation at combined evaluation/save intervals, retaining only the **five best checkpoints** based on evaluation metrics (perplexity/loss) and deleting the worst when an improved checkpoint is saved.
10. **Proportional Multi-Dataset Mixing**: Supports configuring multi-dataset mixtures (e.g. Wikipedia 50%, OSCAR 30%, C4 20%) with percentage weights via inline YAML or standalone `--data-mix` configs with deterministic interleaving.
11. **Supervised Fine-Tuning (SFT)**: Turn-based dialogue fine-tuning (`train/sft.py`) utilizing Gemma 3 turn markers (`<start_of_turn>`, `<end_of_turn>`) and prompt loss masking (`labels = -100` on user tokens).
12. **Direct Preference Optimization (DPO)**: Alignment and RL optimization (`train/dpo.py`) using frozen reference-model log-probabilities and $\beta$ reward margins.
13. **Interactive Streaming Inference**: Terminal-based chat interface (`inference.py`) with streaming decoding and configurable sampling controls (`do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`).

---

## Directory Structure

```text
french_gemma/
├── configs/
│   ├── mlx_config.yaml         # macOS (MPS, bfloat16, pin_memory disabled)
│   ├── amd_config.yaml         # AMD ROCm (CUDA, bfloat16, max_seq_len 4096)
│   ├── nvidia_config.yaml      # Nvidia Jetson Orin Nano (CUDA, float16)
│   ├── sft_config.yaml         # SFT turn-based fine-tuning config (save_dir: ./checkpoints/sft)
│   └── dpo_config.yaml         # DPO preference alignment config (save_dir: ./checkpoints/dpo)
├── hooks/
│   └── README.md               # Developer environment commands and rules
├── train/
│   ├── __init__.py             # Package exports (BaseTrainer, TrainingFactory, strategies)
│   ├── base.py                 # Abstract base classes, protocols, and StrategyType unions
│   ├── builder.py              # ModularTrainer and dependency injection TrainingFactory
│   ├── cli.py                  # Dynamic CLI argument parser & TrainingConfig plumbing
│   ├── pretrain.py             # Pretraining CLI entrypoint (python -m train.pretrain)
│   ├── sft.py                  # Supervised Fine-Tuning entrypoint (python -m train.sft)
│   ├── dpo.py                  # Direct Preference Optimization entrypoint (python -m train.dpo)
│   └── strategies/
│       ├── base.py             # AbstractTrainingStrategy base definition
│       ├── pretrain.py         # Causal Language Model pretraining strategy
│       ├── sft.py              # SFT prompt-masked loss strategy
│       └── dpo.py              # DPO reference-model preference loss strategy
├── scripts/
│   ├── run_ddp.sh              # Multi-GPU launcher helper
│   └── training_example.py     # Standalone simple training example script
├── src/
│   ├── config.py               # YAML configuration parser, data mix schemas, DPO configs
│   ├── dataset.py              # Tokenizer with Gemma 3 turn tokens, packing, and DataLoaders
│   ├── sft_dataset.py          # Conversational SFT dataset and prompt masking
│   ├── dpo_dataset.py          # Preference DPO dataset and sequence log-prob extraction
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

### Modality-Specific Checkpoint Routing & `--save-dir`
Checkpoints are automatically organized into dedicated subdirectories under `./checkpoints`:
- Pretraining (`train.pretrain`): `./checkpoints/pretrain`
- SFT (`train.sft`): `./checkpoints/sft`
- DPO (`train.dpo`): `./checkpoints/dpo`

You can override the target directory on any entrypoint using `--save-dir` (or `--output-dir`):
```bash
source .venv/bin/activate && python -m train.pretrain --config configs/nvidia_config.yaml --save-dir ./checkpoints/my_run
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

## Supervised Fine-Tuning (SFT) (`python -m train.sft`)

Run turn-based conversational fine-tuning on dialogues using the standard Gemma 3 turn format (`<start_of_turn>`, `<end_of_turn>`) with prompt loss masking:

```bash
# Run SFT continuing from a local pretrained model checkpoint
source .venv/bin/activate && python -m train.sft --config configs/sft_config.yaml --pretrained-model-path ./checkpoints/pretrain/best_model --max-steps 1000 --learning-rate 2e-5
```

### Pretrained Checkpoint Continuation & Fallback Warning
*   **Local Pretrained Checkpoint**: Pass `--pretrained-model-path <path>` (or set `pretrained_model_path` in your config) pointing to a local pretraining checkpoint directory or `.pt`/`.bin` file.
*   **Base HuggingFace Fallback**: If no local checkpoint is provided, SFT logs a warning:
    ```text
    WARNING [train.sft] No local pretrained checkpoint provided via --pretrained-model-path. Defaulting to base Gemma 3 checkpoint from HuggingFace: 'google/gemma-3-270m-it'.
    ```

### Default SFT Dataset Mixture (`configs/sft_config.yaml`)
By default, SFT trains on a curated mixture of high-quality French instruction and conversational datasets:
```text
├── Data Mix:
│    ├── 40% OpenLLM-France/Luciole SFT + Claire-Dialogue (Conversational dynamics)
│    ├── 30% ministere-culture (Organic user intents)
│    └── 30% FQuAD (Task, reasoning & Q&A)
```

Configured in `configs/sft_config.yaml`:
```yaml
data_mix:
  - dataset_path: "OpenLLM-France/Luciole-PostTraining-Dataset-1.1"
    dataset_name: "sft_instruct"
    percentage: 40.0
    split: "croissant_aligned_instruct"
  - dataset_path: "ministere-culture/comparia-votes"
    percentage: 30.0
    split: "train"
  - dataset_path: "almanach/fquad"
    percentage: 30.0
    split: "train"
```

### Turn Format & Loss Masking
Conversations are tokenized as:
```text
<bos><start_of_turn>user\nBonjour, comment t'appelles-tu ?<end_of_turn>\n<start_of_turn>model\nBonjour, je suis FrenchGemma, un LLM entraîné en français.<end_of_turn>\n
```
Prompt tokens (user and system turns) are automatically assigned a label of `-100`, ensuring cross-entropy optimization gradients are computed strictly over the model response tokens.

---

## Direct Preference Optimization (DPO) (`python -m train.dpo`)

Align French Gemma models with human preferences using Direct Preference Optimization (DPO):

```bash
# Run DPO alignment with beta=0.1
source .venv/bin/activate && python -m train.dpo --config configs/mlx_config.yaml --dpo-beta 0.1 --max-steps 500 --learning-rate 5e-6
```

### DPO Loss & Reward Metrics
DPO optimizes the policy $\pi_\theta$ against a frozen reference model $\pi_{\text{ref}}$:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
During training, the trainer monitors `reward_accuracy` and `reward_margin` metrics to track alignment progress.

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
