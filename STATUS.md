# Genesis Manthan — Project Status & Session Memory

> Last updated: 2026-04-08  
> Model target: `shahansha/Manthan-1.5B` on HuggingFace Hub  
> Base model: `Qwen/Qwen2.5-1.5B-Instruct`  
> Hardware: **Local = RTX 3500 Ada (12.9 GB VRAM)** — no Kaggle dependency

---

## ✅ Completed

### Infrastructure
- [x] Full project scaffold (`src/`, `tests/`, `configs/`, `docs/`, `notebooks/`)
- [x] All 12 Python source modules + smoke tests passing
- [x] **40/40 pytest unit tests passing** (`pytest tests/`)
- [x] PyTorch CUDA confirmed — `torch 2.6.0+cu124`, RTX 3500 Ada 12.9 GB VRAM
- [x] Unsloth broken on Windows (triton AttrsDescriptor) — graceful fallback to peft + bitsandbytes
- [x] `sys.path` guards on all entrypoint scripts
- [x] Gradient checkpointing enabled in non-Unsloth path (VRAM savings)

### Data
- [x] `src/data/build_local_dataset.py` — **100 diverse, locally-executed, verified traces**
  - 25 math, 25 factual_qa, 25 code_debug, 25 data_analysis
  - All 100/100 executed and verified against real Python output
  - No API key needed — runs entirely offline
- [x] `src/data/generate_synthetic.py` — Gemini/Claude/OpenAI API generator (for scale-up later)
- [x] `src/data/format_dataset.py` — converts raw traces → ChatML HuggingFace Dataset
- [x] Dataset formatted: **90 train / 10 eval** → `data/processed/manthan_dataset`  
  - p50 tokens: 206, max tokens: 517
- [x] `src/data/reward_dataset.py` — 500 GRPO reward samples (200 GSM8K + 150 MBPP + 150 TriviaQA)

### Training
- [x] `src/training/sft_train.py` — Phase 1 SFT with QLoRA, TRL 1.0 SFTConfig API
- [x] `src/training/grpo_train.py` — Phase 2 GRPO with tool-execution rewards
- [x] `configs/sft_local_gpu.yaml` — **Full local GPU SFT config** (5 epochs, RTX 3500 Ada)
- [x] `configs/grpo_local_gpu.yaml` — **Full local GPU GRPO config** (3 epochs, RTX 3500 Ada)
- [x] GRPO reward: non-zero signal (0.35–0.6 range) confirmed with local 5-step test

### Inference
- [x] `src/inference/budget_forcing.py` — BudgetForcingProcessor (LogitsProcessor subclass)
- [x] `generate_with_budget_forcing()` verified end-to-end

### Evaluation  
- [x] **Fallback numeric scoring in benchmark_gsm8k.py** — reports both:
  - `accuracy` (fair): counts verbal numeric answers even without `<final_answer>` tag
  - `accuracy_strict`: only counts `<final_answer>` tag answers
- [x] `src/eval/benchmark_mbpp.py` — handles both `text` and `prompt` field names

### Pipeline
- [x] `run_local_pipeline.ps1` — one-command local pipeline with `-Skip*` flags
  - New flags: `-SkipBuildLocalDataset`, `-SftConfig`, `-GrpoConfig`, `-EvalModel`
  - Defaults to `sft_local_gpu.yaml` and `grpo_local_gpu.yaml`

---

## 🔲 Remaining Work — All Local GPU

### Step 1 — Full SFT Training (~1-2 hours on RTX 3500 Ada)

```powershell
.\.venv\Scripts\python.exe src/training/sft_train.py --config configs/sft_local_gpu.yaml
```

Expected: ~5 epochs on 90 samples, loss < 0.5 at convergence  
Output: `outputs/sft_local_gpu/`

### Step 2 — Full GRPO Training (~2-4 hours)

```powershell
.\.venv\Scripts\python.exe src/training/grpo_train.py --config configs/grpo_local_gpu.yaml
```

Expected: reward improving from ~0.4 to >0.7 over training  
Output: `outputs/grpo_local_gpu/`

### Step 3 — Evaluate

```powershell
# GSM8K: 100 samples, full eval
.\.venv\Scripts\python.exe src/eval/benchmark_gsm8k.py `
  --model outputs/grpo_local_gpu --n-samples 100 `
  --output outputs/eval/gsm8k_full.json

# MBPP: 50 samples
.\.venv\Scripts\python.exe src/eval/benchmark_mbpp.py `
  --model outputs/grpo_local_gpu --n-samples 50 `
  --output outputs/eval/mbpp_full.json
```

Targets: GSM8K accuracy (fallback) > 40%, tool_call_parsability > 70%

### Step 4 — Scale SFT data with API (optional)

```powershell
# Generate 200+ more traces with Gemini (requires GEMINI_API_KEY in .env)
.\.venv\Scripts\python.exe src/data/generate_synthetic.py `
  --n-samples 200 --output data/raw/synthetic_traces.jsonl `
  --provider gemini --model gemini-2.5-flash --request-delay 13
```

### Step 5 — Push to HuggingFace Hub (optional)

Set `HF_TOKEN` in `.env`, then set `push_to_hub: true` in configs.

---

## ⚠️ Known Issues / Notes

- **Unsloth**: Broken on Windows due to `triton.compiler.compiler.AttrsDescriptor` conflict.  
  Fallback to standard peft + bitsandbytes works correctly.
- **GSM8K eval accuracy_strict = 0%**: Model answers verbally (no `<final_answer>` tag).  
  `accuracy` (fallback) is the meaningful metric during early training iterations.
- **fp16 disabled**: `fp16: false` in all configs — AMP GradScaler conflicts with 4-bit bnb optimizer.
- **Kaggle configs**: `configs/sft_config.yaml` and `configs/grpo_config.yaml` kept for reference;  
  use `sft_local_gpu.yaml` and `grpo_local_gpu.yaml` for local training.

---

## Key Commands Reference

```powershell
# Full one-command local pipeline (builds data, trains, evals)
powershell -ExecutionPolicy Bypass -File .\run_local_pipeline.ps1

# Skip already-done steps
powershell -ExecutionPolicy Bypass -File .\run_local_pipeline.ps1 `
  -SkipBuildLocalDataset -SkipFormatDataset -SkipRewardDataset `
  -SkipSft -EvalSamples 20

# Run all tests
.\.venv\Scripts\python.exe -m pytest tests/ -q

# Smoke test individual scripts
.\.venv\Scripts\python.exe src/training/sft_train.py --smoke-test
.\.venv\Scripts\python.exe src/training/grpo_train.py --smoke-test
.\.venv\Scripts\python.exe src/data/build_local_dataset.py --smoke-test
```

---

## File / Output Reference

| Path | Description |
|---|---|
| `data/raw/local_traces.jsonl` | 100 diverse, locally-executed traces (Math/Code/QA/Analysis) |
| `data/raw/synthetic_traces.jsonl` | Gemini-generated traces (API-based, optional scale-up) |
| `data/processed/manthan_dataset` | ChatML HuggingFace Dataset (90 train / 10 eval) |
| `data/processed/reward_dataset.jsonl` | 500 GRPO reward samples with ground truth |  
| `outputs/sft_local_gpu/` | ← **Full SFT checkpoint** (after Step 1) |
| `outputs/grpo_local_gpu/` | ← **Full GRPO checkpoint** (after Step 2) |
| `outputs/sft_local_test/` | Quick 10-step smoke-test SFT adapter |
| `outputs/grpo_local_test/` | Quick 5-step smoke-test GRPO adapter |
| `configs/sft_local_gpu.yaml` | Local GPU SFT config (5 epochs, no Hub push) |
| `configs/grpo_local_gpu.yaml` | Local GPU GRPO config (3 epochs, no Hub push) |


---

## ✅ Completed

### Infrastructure
- [x] Full project scaffold created (`src/`, `tests/`, `configs/`, `docs/`, `notebooks/`)
- [x] All 12 Python source modules written and passing smoke tests
- [x] 38/38 pytest unit tests passing (`pytest tests/`)
- [x] PyTorch CUDA confirmed working — `torch 2.6.0+cu124` in `.venv`
- [x] `trl 1.0.0`, `bitsandbytes 0.49.2`, `huggingface_hub` installed
- [x] Unsloth installed but **broken on Windows** (triton `AttrsDescriptor` conflict) — graceful fallback implemented

### Data
- [x] `src/data/generate_synthetic.py` — Gemini 2.5 Flash as primary provider
- [x] 14 high-quality tool-interaction traces generated → `data/raw/synthetic_traces.jsonl`
- [x] `src/data/format_dataset.py` — converts raw traces → ChatML HuggingFace Dataset
- [x] Dataset formatted: 13 train / 1 eval → `data/processed/manthan_dataset`
  - p50 tokens: 194, max tokens: 404

### Training
- [x] `src/training/sft_train.py` — Phase 1 SFT with QLoRA, TRL 1.0 `SFTConfig` API
- [x] `configs/sft_local_test.yaml` — 10-step local test config
- [x] **10-step GPU SFT training completed** on RTX 3500 Ada
  - loss: 2.655 → 1.492 over 10 steps
  - mean token accuracy at step 10: 73.1%
  - VRAM used: ~1.2GB (model in 4-bit NF4)
  - Adapter saved to: `outputs/sft_local_test/`

### Inference
- [x] `src/inference/budget_forcing.py` — `BudgetForcingProcessor` (LogitsProcessor subclass)
- [x] `generate_with_budget_forcing()` tested end-to-end
- [x] Correctly answered "17 × 23 = 391" with `min_calls=1, max_calls=3`

---

## 🔲 Remaining Work

### Phase 1 — Scale SFT Data and Train on Kaggle
- [ ] Generate 200+ training samples using `gemini-2.5-flash`
  ```powershell
  .\.venv\Scripts\python.exe src/data/generate_synthetic.py `
    --n-samples 200 --output data/raw/synthetic_traces.jsonl `
    --provider gemini --model gemini-2.5-flash --request-delay 13
  ```
  > Rate limits: 5 RPM / ~1500 RPD free tier. Use `--request-delay 13` (safe for 4.5 req/min).  
  > **Do NOT use `gemini-2.5-flash-lite`** — only 20 RPD free tier, exhausted in 14 samples.

- [ ] Re-format dataset with more samples
  ```powershell
  .\.venv\Scripts\python.exe src/data/format_dataset.py `
    --input data/raw/synthetic_traces.jsonl `
    --output data/processed/manthan_dataset
  ```

- [ ] Run full SFT on Kaggle (3 epochs, ~3 hours on T4)
  - Upload `src/training/sft_train.py`, `configs/sft_config.yaml`, dataset
  - Set Kaggle secret: `HF_TOKEN`
  - After training: adapter auto-pushed to `shahansha/Manthan-1.5B`
  - Use `configs/sft_config.yaml` (not `sft_local_test.yaml`)

### Phase 2 — GRPO Reinforcement Learning
- [ ] Prepare GRPO reward dataset (GSM8K / MBPP subsets)
  ```powershell
  .\.venv\Scripts\python.exe src/data/reward_dataset.py
  ```
- [ ] Run `src/training/grpo_train.py` on Kaggle
  - Reward functions in `src/training/reward_functions.py` (all tested, 38/38 passing):
    - `tool_execution_reward` — JSON parsability + execution success
    - `answer_correctness_reward` — exact/fuzzy match
    - `format_reward` — presence of `<tool_call>` block
  - Config: `configs/grpo_config.yaml`
  - Sessions max 8–10 hours (within 30hrs/week Kaggle cap)
- [ ] Verify GRPO improves tool call JSON parsability (target: >85%)

### Phase 3 — Budget Forcing RL
- [ ] Implement budget forcing training loop (currently only inference-time forcing exists)
- [ ] Inject "Wait" token when model produces `<final_answer>` before `min_tool_calls`
- [ ] Target: avg tool calls before answer = 1.5–3.0

### Evaluation
- [ ] `src/eval/benchmark_gsm8k.py` — run pass@1 on GSM8K (target: >65%)
- [ ] `src/eval/benchmark_mbpp.py` — run pass@1 on MBPP (target: >50%)
- [ ] `src/eval/tool_success_rate.py` — tool-execution success rate metric

### Inference & Demo
- [ ] `src/inference/smolagents_integration.py` — wrap as smolagents CodeAgent
- [ ] `src/inference/demo.py` — Gradio split-screen demo

---

## ⚠️ Known Issues & Workarounds

| Issue | Workaround |
|---|---|
| Unsloth broken on Windows (triton `AttrsDescriptor`) | `_load_model_with_fallback()` in `sft_train.py` → falls back to `AutoModelForCausalLM + BitsAndBytesConfig + peft` |
| TRL 1.0 API: `SFTTrainer` no longer accepts `tokenizer=` | Use `processing_class=tokenizer` |
| TRL 1.0 API: `TrainingArguments` replaced by `SFTConfig` | Import `from trl import SFTConfig` and pass `dataset_text_field`, `max_length` inside it |
| AMP fp16 conflicts with bitsandbytes 4-bit (bfloat16 internal tensors) | Set `fp16: false` in training config; bnb handles efficiency internally |
| `torch._amp_foreach_non_finite_check_and_unscale_cuda` BFloat16 error | Fixed by disabling `fp16` in `SFTConfig` |
| Windows cp1252 UnicodeEncodeError on `✓` char | `format_dataset.py` wraps stdout in UTF-8 `io.TextIOWrapper` |
| `grpo_train.py` ModuleNotFoundError for `src` | `sys.path.insert(0, _PROJECT_ROOT)` added at module top |
| `gemini-2.5-flash` 503 errors (high demand) | Parse `retryDelay` from 429 responses; `max_retries=5`, sleep accordingly |

---

## 🔧 Environment

```
Python:        3.10+
PyTorch:       2.6.0+cu124  (CUDA confirmed)
TRL:           1.0.0
bitsandbytes:  0.49.2
transformers:  5.5.0
unsloth:       installed but broken on Windows
Gemini model:  gemini-2.5-flash (default, ~1500 RPD free)
GPU local:     NVIDIA RTX 3500 Ada, 12.9GB VRAM, CUDA 12.4
GPU Kaggle:    T4, 16GB VRAM (FP16 only — never bfloat16)
```

Key env vars needed:
- `GEMINI_API_KEY` — in `.env` file
- `HF_TOKEN` — needed for Kaggle Hub push (set as Kaggle secret)

---

## 📁 Key File Locations

| File | Purpose |
|---|---|
| `data/raw/synthetic_traces.jsonl` | 14 raw Gemini-generated traces |
| `data/processed/manthan_dataset` | Formatted HF Dataset (13 train / 1 eval) |
| `outputs/sft_local_test/` | Fine-tuned LoRA adapter (10-step test run) |
| `configs/sft_local_test.yaml` | Local test config (max_steps=10, fp16=false) |
| `configs/sft_config.yaml` | Full Kaggle SFT config |
| `configs/grpo_config.yaml` | GRPO config |
| `src/training/sft_train.py` | Phase 1 training script |
| `src/training/grpo_train.py` | Phase 2 training script |
| `src/training/reward_functions.py` | Composable reward functions (tested) |
| `src/inference/budget_forcing.py` | BudgetForcingProcessor + generate helper |
| `src/data/generate_synthetic.py` | Gemini data generation |
| `src/data/format_dataset.py` | ChatML formatting pipeline |

---

## 🎯 Evaluation Targets

| Metric | Baseline (Qwen2.5-1.5B base) | Target (Manthan-1.5B) |
|---|---|---|
| GSM8K (tool-augmented) | ~45% | >65% |
| MBPP pass@1 | ~35% | >50% |
| Tool call JSON parsability | ~7% | >85% |
| smolagents CodeAgent success | untested | >70% |
| Avg tool calls before answer | N/A | 1.5–3.0 |
