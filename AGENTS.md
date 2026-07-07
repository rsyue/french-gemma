# Agent Instructions & Guidelines (AGENTS.md)

Welcome, agent! This document contains critical guidelines, architecture patterns, and conventions for working on the French Gemma 3 pretraining repository. Always adhere to these instructions.

---

## 1. Environment & Commands Enforcements
*   **Virtual Environment**: You **MUST** run all python/pytest/ruff commands using `source .venv/bin/activate && <command>`.
*   **Testing**: Before submitting or presenting changes, ensure all tests pass:
    ```bash
    source .venv/bin/activate && pytest tests/
    ```
*   **Linting**: Ensure code conforms to ruff lint standards before finishing your turn:
    ```bash
    source .venv/bin/activate && ruff check .
    ```

---

## 2. Core Repository Architecture
*   **`src/config.py`**: Custom config loader/validator from YAML (`TrainingConfig`).
*   **`src/dataset.py`**: Custom tokenizer training (`ByteLevelBPETokenizer`), flat token packing with a sliding window stride of 50 tokens, and `DataLoader` creation. Uses **right padding** (`padding_side = "right"`) for causal decoder training.
*   **`src/model.py`**: Wraps the base `Gemma3` model initialized from a blank configuration, adding a PyTorch-native `nn.Linear` LM Head. Incorporates **Gaussian embedding noise** (regulated by `embedding_noise_std`) during training.
*   **`src/scheduler.py`**: Handles linear warmup + cosine annealing with warm restarts, and `FreezeManager` layer freezing schedules.
*   **`src/trainer.py`**: Pretraining trainer loop driving optimization, evaluation metrics (loss & perplexity), text generation, and TensorBoard logging. Retains the **top 3 best checkpoints** based on perplexity.

---

## 3. Engineering Best Practices
Always follow these workflows:
1.  **Test-Driven Development (TDD)**: When adding features or fixing bugs, write unit/integration tests first in `tests/` and verify they fail before writing implementation code.
2.  **Unslopping Code**: Remove all temporary debugging print statements, commented-out dead code blocks, or narrator comments that merely describe code syntax. Keep the codebase clean.
3.  **Phased Implementation**: Break down tasks into distinct phases. Work on one major feature per phase. Ask the user for explicit approval at the end of each phase. Commit each approved phase as a **single Git commit**.
4.  **No Warnings**: Ensure that PyTorch or other libraries do not trigger warnings during training or tests (e.g. correct scheduler step ordering).
