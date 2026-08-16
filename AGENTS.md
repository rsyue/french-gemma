# Agent Instructions & Guidelines (AGENTS.md)

Welcome, agent! This document contains critical guidelines, architecture patterns, and conventions for working on the French Gemma 3 pretraining repository. Always adhere to these instructions.

---

## 1. Environment & Commands Enforcements
*   **Virtual Environment**: You **MUST** run all python/pytest/ruff/mypy commands using `source .venv/bin/activate && <command>`.
*   **Pip Commands**: Any pip command must be run as `uv pip` (e.g., `source .venv/bin/activate && uv pip ...`).
*   **Testing**: Before submitting or presenting changes, ensure all tests pass:
    ```bash
    source .venv/bin/activate && pytest tests/
    ```
*   **Linting & Formatting**: Ensure code conforms to ruff lint standards before finishing your turn:
    ```bash
    source .venv/bin/activate && ruff check .
    ```
*   **Type Checking**: Ensure code has no static type errors before finishing your turn:
    ```bash
    source .venv/bin/activate && mypy .
    ```

---

## 2. Core Repository Architecture
*   **`src/config.py`**: Custom config loader/validator from YAML (`TrainingConfig`), including multi-dataset mixes (`DatasetMixEntry`, `parse_data_mix_config`), DPO parameters (`dpo_beta`, `dpo_label_smoothing`), and distributed parameters like `dist_timeout_seconds`.
*   **`src/dataset.py`**: Custom tokenizer training (`ByteLevelBPETokenizer`) with Gemma 3 turn tokens (`<start_of_turn>`, `<end_of_turn>`) and Jinja chat template, multi-dataset proportional loading and mixing (`load_dataset_mix`), flat token packing with sliding window, atomic cache generation with `.ready` sentinels, and `DataLoader` creation.
*   **`src/sft_dataset.py`**: Turn-based conversational dataset (`SFTDataset`), collation, and prompt loss masking (`labels = -100` on user/system prompt tokens) for supervised fine-tuning.
*   **`src/dpo_dataset.py`**: Preference dataset (`DPODataset`), collation, and token log-probability extractor (`get_batch_logps`) for Direct Preference Optimization.
*   **`src/model.py`**: Wraps the base `Gemma3` model initialized from a blank configuration, adding a PyTorch-native `nn.Linear` LM Head. Incorporates **Gaussian embedding noise** (regulated by `embedding_noise_std`) during training.
*   **`src/scheduler.py`**: Handles linear warmup + cosine annealing with warm restarts, and `FreezeManager` layer freezing schedules.
*   **`src/trainer.py`**: Pretraining trainer loop driving optimization, evaluation metrics (loss & perplexity), text generation, and TensorBoard logging. Retains the **top 3 best checkpoints** based on perplexity and training loss. Implements process-rank checks (`self.is_main_process`) and validation loss all-reduce for DDP synchronization.
*   **`train/strategies/`**: Pluggable training strategy hierarchy including `PretrainStrategy` (causal LM loss), `SFTStrategy` (prompt-masked cross-entropy), and `DPOStrategy` (reference-model log-ratio preference loss).
*   **`train/pretrain.py`, `train/sft.py`, `train/dpo.py` & `scripts/run_ddp.sh`**: Configurable training entrypoints for pretraining, supervised fine-tuning, and direct preference alignment.
*   **`inference.py`**: Interactive CLI chat interface and text streamer supporting configurable sampling parameters (`do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`).

---

## 3. Engineering Best Practices
Always follow these workflows:
1.  **Test-Driven Development (TDD)**: When adding features or fixing bugs, write unit/integration tests first in `tests/` and verify they fail before writing implementation code.
2.  **Unslopping Code**: Remove all temporary debugging print statements, commented-out dead code blocks, or narrator comments that merely describe code syntax. Keep the codebase clean and utilize logger-based messaging instead of standard print statements.
3.  **Phased Implementation**: Break down tasks into distinct phases. Work on one major feature per phase. Ask the user for explicit approval at the end of each phase. Commit each approved phase as a **single Git commit**.
4.  **No Warnings**: Ensure that PyTorch or other libraries do not trigger warnings during training or tests (e.g. correct scheduler step ordering).
