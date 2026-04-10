# Genesis Manthan — GitHub Copilot Agent Instructions

## Project Identity

You are building **Genesis Manthan (Manthan-1.5B)** — the first open small language model that reasons through **tool interaction** instead of chain-of-thought. This is a solo-developer ML research project by Shahansha Shaik under the **Genesis AGI** brand.

- **HuggingFace target**: `shahansha/Manthan-1.5B`
- **Base model**: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`  
- **Training**: QLoRA + SFT (Phase 1) → GRPO with tool-execution rewards (Phase 2) → Budget Forcing (Phase 3)
- **Target hardware**: Kaggle T4 GPU (16GB VRAM, FP16 only — no bfloat16)

---

## Coding Conventions

### Language & Style
- **Python 3.10+** throughout
- Type hints on all function signatures
- Docstrings only on public functions and classes
- Use `dataclasses` or `pydantic` for config objects, never raw dicts
- Prefer `pathlib.Path` over `os.path`
- All training scripts must work **both locally (CPU smoke test)** and **on Kaggle T4**

### Hardware Portability Rule
Every script that loads a model must support a `--smoke-test` flag that:
- Loads only the **tokenizer** (no model weights)
- Runs a forward pass on dummy data
- Exits without error on CPU in < 10 seconds

```python
# Pattern to follow everywhere:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    # tokenizer only, CPU, no training
    ...
```

### Kaggle / Colab Compatibility
- **Never use bfloat16** — T4 does not support it. Always use `torch.float16`
- Always set `use_gradient_checkpointing="unsloth"` in FastLanguageModel
- Read secrets via `os.environ.get("HF_TOKEN")` — never hardcode tokens
- Use `os.environ.get("KAGGLE_KERNEL_RUN_TYPE")` to detect Kaggle environment
- Notebook cells must be idempotent (re-runnable without errors)

---

## Architecture Decisions

### Data Format
All training data uses **ChatML format** with custom tool roles:
```
<|im_start|>system
You are an agent that solves problems by calling tools.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "python_repl", "arguments": {"code": "..."}}</tool_call><|im_end|>
<|im_start|>tool
{"result": "...", "success": true}<|im_end|>
<|im_start|>assistant
{final_answer}<|im_end|>
```

### Reward Function Structure (GRPO)
Reward functions must return a float in [0.0, 1.0]:
- `tool_execution_reward`: 0.5 if tool call parses as valid JSON, +0.5 if execution succeeds
- `answer_correctness_reward`: 0.0–1.0 based on exact or fuzzy match
- `format_reward`: 0.1 if output contains at least one `<tool_call>` block

Never combine rewards inside a single function — keep them composable.

### Budget Forcing
- Inject `"Wait"` token when model produces `<final_answer>` before minimum tool calls
- Minimum tool calls threshold: configurable, default = 1
- Maximum budget: configurable, default = 5 tool calls per problem
- Implement as a `LogitsProcessor` subclass, not post-processing

---

## File Responsibilities

| File | Purpose |
|---|---|
| `src/data/generate_synthetic.py` | Call Claude/GPT API to generate tool-interaction traces |
| `src/data/format_dataset.py` | Convert raw traces → ChatML HuggingFace Dataset |
| `src/data/reward_dataset.py` | Curate GSM8K/MBPP/TriviaQA for GRPO reward checking |
| `src/training/sft_train.py` | Phase 1 SFT with QLoRA via Unsloth |
| `src/training/grpo_train.py` | Phase 2 GRPO via TRL GRPOTrainer |
| `src/training/reward_functions.py` | Composable reward signal functions |
| `src/inference/budget_forcing.py` | Budget forcing LogitsProcessor |
| `src/inference/smolagents_integration.py` | Wrap Manthan as smolagents CodeAgent |
| `src/inference/demo.py` | Gradio split-screen demo |
| `src/eval/benchmark_gsm8k.py` | GSM8K pass@1 evaluator |
| `src/eval/benchmark_mbpp.py` | MBPP pass@1 evaluator |
| `src/eval/tool_success_rate.py` | Custom tool-execution success metric |

---

## Dependencies

### Core Training (Kaggle)
```
unsloth[kaggle-new]  # or unsloth[colab-new] for Colab
trl>=0.9.0
transformers>=4.45.0
datasets>=2.20.0
peft>=0.12.0
accelerate>=0.33.0
torch>=2.3.0  # installed by Kaggle
```

### Local Dev (CPU only)
```
transformers>=4.45.0
datasets>=2.20.0
tokenizers
anthropic  # for synthetic data generation
openai     # alternative for synthetic data generation
gradio>=4.0.0
evaluate
```

### Evaluation
```
evaluate
rouge-score
sacrebleu
```

---

## Workflow: Local First, Then Kaggle

### Phase: Local (VS Code)
1. **Develop and test all code on CPU** using `--smoke-test`
2. **Generate synthetic dataset** using free API keys (no GPU)
3. **Format and validate dataset** — check ChatML structure, token counts
4. **Unit test reward functions** with mock tool outputs
5. **Validate notebook cells** are correct Python before uploading

### Phase: Kaggle
1. Upload notebooks via Kaggle API or web UI
2. Set secrets: `HF_TOKEN`, `ANTHROPIC_API_KEY` (optional)
3. Run SFT first (~3 hours), checkpoint uploaded to HuggingFace Hub
4. Run GRPO in sessions of max 8–10 hours (within 30hrs/week cap)
5. Download final checkpoint for local evaluation

---

## Key Research References

- **Foundational paper**: Rainone et al., arXiv:2507.05065 — Tool-mediated reasoning in small models
- **Budget forcing**: arXiv:2510.21398 — Budget forcing for 1.5B models (1.5K samples, token efficiency via RL)
- **s1 codebase**: github.com/simplescaling/s1 — Budget forcing implementation (~50 lines)
- **Unsloth GRPO**: unsloth.ai/blog/grpo — T4-compatible GRPO with QLoRA
- **Training data**: `glaiveai/glaive-function-calling-v2`, `NousResearch/hermes-function-calling-v1`
- **Benchmark data**: GSM8K, MBPP, TriviaQA (all on HuggingFace)

---

## What NOT to Do

- Do NOT use bfloat16 anywhere — T4 will OOM or produce NaN
- Do NOT hardcode HuggingFace tokens, API keys, or secrets
- Do NOT train on full-precision — always 4-bit QLoRA via Unsloth
- Do NOT write monolithic training scripts — keep data, training, eval separate
- Do NOT add features beyond the three training phases without explicit request
- Do NOT create new abstractions for one-off operations
- Do NOT add chain-of-thought verbal reasoning to the training data — this is tool-mediated only

---

## Evaluation Targets (Success Criteria)

| Metric | Baseline (Qwen2.5-1.5B base) | Target (Manthan-1.5B) |
|---|---|---|
| GSM8K (tool-augmented) | ~45% | >65% |
| MBPP pass@1 | ~35% | >50% |
| Tool call JSON parsability | ~7% | >85% |
| smolagents CodeAgent success | untested | >70% on HF agent benchmark |
| Avg tool calls before answer | N/A | 1.5–3.0 |
