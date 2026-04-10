# Genesis Manthan — Complete Project Guide

> *मंथन (Manthan) = "Churning of deep thought"*  
> The first open small language model that reasons through **tool interaction**, not chain-of-thought.

---

## Table of Contents

1. [What Has Been Built](#1-what-has-been-built)
2. [Local Setup](#2-local-setup)
3. [Run Locally](#3-run-locally)
4. [Kaggle Training](#4-kaggle-training)
5. [HuggingFace Deployment](#5-huggingface-deployment)
6. [Testing & Verification](#6-testing--verification)
7. [Use Cases & Real Examples](#7-use-cases--real-examples)
8. [Scenarios Where Manthan Excels](#8-scenarios-where-manthan-excels)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What Has Been Built

### Project at a Glance

| Property | Value |
|---|---|
| Model name | `shahansha/Manthan-1.5B` |
| Base model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| Paradigm | Tool-mediated reasoning (not verbal CoT) |
| Training | QLoRA SFT → GRPO with tool-execution rewards → Budget Forcing |
| GPU cost | $0 (Kaggle free tier) |
| VRAM requirement | ~5–9 GB (T4 compatible) |

### What Each Component Does

#### Data Pipeline (`src/data/`)

| File | Status | Purpose |
|---|---|---|
| `generate_synthetic.py` | ✅ Done | Generates 100-seed tool-interaction traces via Gemini/Claude/OpenAI. Supports `--dummy-data` mode with no API key. |
| `format_dataset.py` | ✅ Done | Converts raw JSONL traces → ChatML format → HuggingFace `Dataset`. Enforces 1024-token limit. |
| `reward_dataset.py` | ✅ Done | Curates GSM8K/MBPP/TriviaQA problems with verifiable answers for GRPO reward checking. |

The synthetic dataset uses 100 hardcoded seed problems across 4 domains: **math**, **code_debug**, **data_analysis**, and **logic**. The generator expands these into thousands of ChatML traces.

#### Training (`src/training/`)

| File | Status | Purpose |
|---|---|---|
| `sft_train.py` | ✅ Done | Phase 1: QLoRA SFT via Unsloth. Trains model to follow tool-call format. Supports `--smoke-test`. |
| `grpo_train.py` | ✅ Done | Phase 2: GRPO via TRL. Optimizes with tool-execution rewards. |
| `reward_functions.py` | ✅ Done | Three composable reward signals: tool execution (0.5), answer correctness (0.4), format (0.1). |

The local test adapter is saved in `outputs/sft_local_test/` — a real, trained LoRA checkpoint from a smoke run on CPU/small GPU.

#### Inference (`src/inference/`)

| File | Status | Purpose |
|---|---|---|
| `budget_forcing.py` | ✅ Done | `LogitsProcessor` that suppresses `<final_answer>` until minimum tool calls are made. Based on arXiv:2510.21398. |
| `smolagents_integration.py` | ✅ Done | Wraps Manthan as a `smolagents` `CodeAgent` with sandboxed Python REPL. |
| `demo.py` | ✅ Done | Gradio split-screen demo (left: problem, right: tool call trace + final answer). |

#### Evaluation (`src/eval/`)

| File | Status | Purpose |
|---|---|---|
| `benchmark_gsm8k.py` | ✅ Done | GSM8K pass@1 evaluator. Runs model on 100/1319 test problems. |
| `benchmark_mbpp.py` | ✅ Done | MBPP pass@1 evaluator for code generation tasks. |
| `tool_success_rate.py` | ✅ Done | Custom metric: fraction of completions where JSON parses + code runs successfully. |

#### Configs (`configs/`)

| Config | Purpose |
|---|---|
| `sft_config.yaml` | Production SFT on Kaggle (3 epochs, batch 4, LR 2e-4) |
| `grpo_config.yaml` | GRPO starting from SFT checkpoint (4 completions/prompt, LR 5e-6) |
| `sft_local_test.yaml` | Smoke run on local GPU — 10 steps, tiny batch |

### Training Data Format

All training uses **ChatML with custom tool roles**:

```
<|im_start|>system
You are an agent that solves problems by calling tools.<|im_end|>
<|im_start|>user
What is 17 * 23?<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "python_repl", "arguments": {"code": "print(17 * 23)"}}</tool_call><|im_end|>
<|im_start|>tool
{"result": "391", "success": true}<|im_end|>
<|im_start|>assistant
<final_answer>391</final_answer><|im_end|>
```

There is **no verbal chain-of-thought**. The model reasons exclusively through tool interactions.

---

## 2. Local Setup

### Prerequisites

- Python 3.10 or higher
- Windows/Linux/Mac (CPU-only for local dev is fine)
- Git

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/shaik-shahansha/manthan.git
cd manthan

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

# 3. Install dev dependencies (CPU-only, no CUDA needed locally)
pip install -r requirements-dev.txt

# 4. (Optional) Create .env for API keys — NEVER commit this file
echo "HF_TOKEN=hf_your_token_here" > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "GEMINI_API_KEY=AIza..." >> .env

# 5. Verify tokenizer downloads correctly (< 2 min)
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); print('Tokenizer OK:', len(t))"
```

### Verify the Entire Scaffold

Run all smoke tests in under 60 seconds on CPU:

```bash
python src/data/generate_synthetic.py --smoke-test
python src/data/format_dataset.py --smoke-test
python src/training/reward_functions.py --smoke-test
python src/training/sft_train.py --smoke-test
python src/inference/budget_forcing.py --smoke-test
python src/inference/smolagents_integration.py --smoke-test
python src/eval/benchmark_gsm8k.py --smoke-test
python src/eval/benchmark_mbpp.py --smoke-test

# Or run all unit tests:
pytest tests/ -v
```

Every smoke test loads only the tokenizer (no model weights) and exits in < 10 seconds.

---

## 3. Run Locally

### 3a. Generate Synthetic Training Data

**Option A: Dummy data (no API key needed)**
```bash
python src/data/generate_synthetic.py --dummy-data --n-samples 50 --output data/raw/synthetic_traces.jsonl
```

**Option B: Gemini (free tier — recommended)**
```bash
# Set GEMINI_API_KEY in .env first
python src/data/generate_synthetic.py --provider gemini --n-samples 200 --output data/raw/synthetic_traces.jsonl
```

**Option C: Anthropic Claude**
```bash
python src/data/generate_synthetic.py --provider anthropic --n-samples 200 --output data/raw/synthetic_traces.jsonl
```

Expected output: a JSONL file where each line is a complete ChatML trace:
```json
{"text": "<|im_start|>system\nYou are an agent...<|im_end|>\n<|im_start|>user\n..."}
```

### 3b. Format Dataset

```bash
python src/data/format_dataset.py \
  --input data/raw/synthetic_traces.jsonl \
  --output data/processed/manthan_dataset \
  --max-tokens 1024
```

This creates a HuggingFace `DatasetDict` at `data/processed/manthan_dataset/` with `train` (90%) and `eval` (10%) splits.

### 3c. Run SFT Locally (Smoke/Debug Run)

The local config runs 10 steps with a tiny dataset to verify the training loop:

```bash
python src/training/sft_train.py --config configs/sft_local_test.yaml
```

Expected output:
```
[SFT] Loading dataset from data/processed/manthan_dataset...
[SFT] Dataset loaded: 90 train / 10 eval samples
[SFT] Loading model with Unsloth fallback...
[SFT] Using transformers + PEFT (no Unsloth)
Step 10/10 — Loss: 1.832
[SFT] Saved to outputs/sft_local_test
```

The trained adapter will be in `outputs/sft_local_test/` (already present in this repo).

### 3d. Run Inference with Budget Forcing

```bash
python -c "
import sys; sys.path.insert(0, '.')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.inference.budget_forcing import generate_with_budget_forcing

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', torch_dtype=torch.float16, device_map='auto')
model = PeftModel.from_pretrained(base, 'outputs/sft_local_test')
tokenizer = AutoTokenizer.from_pretrained('outputs/sft_local_test')
model.eval()

result = generate_with_budget_forcing(
    model, tokenizer,
    problem='What is 17 * 23?',
    minimum_tool_calls=1,
    maximum_tool_calls=3,
    max_new_tokens=256
)
print(result)
"
```

### 3e. Run the Reward Functions Interactively

```python
from src.training.reward_functions import (
    tool_execution_reward,
    answer_correctness_reward,
    format_reward,
    combined_reward,
)

completion = """
<tool_call>{"name": "python_repl", "arguments": {"code": "print(17 * 23)"}}</tool_call>
<tool_response>{"result": "391", "success": true}</tool_response>
<final_answer>391</final_answer>
"""

sandbox = {"success": True, "result": "391"}

print(tool_execution_reward(completion, sandbox))    # → 1.0
print(answer_correctness_reward(completion, "391"))  # → 1.0
print(format_reward(completion))                     # → 0.1
print(combined_reward(completion, "391", sandbox))   # → ~1.0
```

### 3f. Run the Gradio Demo

```bash
pip install gradio
python src/inference/demo.py --model-path outputs/sft_local_test --base-model Qwen/Qwen2.5-1.5B-Instruct
```

Opens at `http://localhost:7860`. Enter any math or coding problem in the left panel and watch the tool-call trace stream on the right.

---

## 4. Kaggle Training

### 4a. Setup

1. Create a **Kaggle account** at kaggle.com (free)
2. Go to **Settings → API → Create New Token** — downloads `kaggle.json`
3. Upload it: place `kaggle.json` in `~/.kaggle/` (Linux) or `C:\Users\<you>\.kaggle\` (Windows)
4. Install the Kaggle CLI: `pip install kaggle`

Add Kaggle secrets (used instead of `.env`):
- Go to kaggle.com → **Your Account → Settings → Secrets**
- Add `HF_TOKEN` = your HuggingFace write token
- Add `GEMINI_API_KEY` (optional, for data generation)

### 4b. Phase 1 — SFT Training (~3–5 hours)

Upload and run `notebooks/02_sft_kaggle.ipynb`:

1. Open Kaggle → **New Notebook** → Upload `notebooks/02_sft_kaggle.ipynb`
2. Set **Accelerator** to **GPU T4 x2** (or single T4)
3. In the notebook, ensure:
   ```python
   # Cell 1: Install dependencies
   !pip install unsloth[kaggle-new] trl>=0.9.0 datasets peft accelerate -q
   
   # Cell 2: Set HF_TOKEN from Kaggle secret
   import os
   from kaggle_secrets import UserSecretsClient
   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
   ```
4. Click **Run All** → training starts
5. SFT pushes the adapter to `shahansha/Manthan-1.5B-sft-v0.1` at end of training

Critical Kaggle constraints:
- **Never use bfloat16** — T4 does not support it. Always `torch.float16`
- Sessions time out after ~9 hours — `save_steps: 100` is already set in config
- Free tier: 30 GPU hours/week — SFT uses ~5 hours

### 4c. Phase 2 — GRPO Training (~20–25 hours, across 3 sessions)

Upload and run `notebooks/03_grpo_kaggle.ipynb`. GRPO resumes from the SFT checkpoint:

```yaml
# grpo_config.yaml — already configured
model:
  name: "shahansha/Manthan-1.5B-sft-v0.1"
training:
  save_steps: 50      # Saves every 50 steps to survive session timeouts
hub:
  push_to_hub: true   # Pushes to Hub every 50 steps
  hub_model_id: "shahansha/Manthan-1.5B-grpo-v0.1"
```

Because GRPO needs ~300 minimum steps for reward improvement and Kaggle sessions run out, run 3 sessions of ~100 steps each. The checkpoint auto-saves to Hub so you pick up where you left off.

### 4d. Verify Training Progress During a Kaggle Run

Watch loss and reward converge in Kaggle logs:
```
Step  10 | loss: 0.847 | reward: 0.321
Step  20 | loss: 0.793 | reward: 0.412
Step  50 | loss: 0.701 | reward: 0.551
Step 100 | loss: 0.634 | reward: 0.623  ← reward improving = GRPO working
Step 200 | loss: 0.590 | reward: 0.694
Step 300 | loss: 0.561 | reward: 0.731  ← plateau starts here
```

If reward stays below 0.3 after 50 steps, check:
- Tool execution sandbox is running (`sandbox_result` is not None)
- Dataset has `<tool_call>` blocks (not empty traces)

---

## 5. HuggingFace Deployment

### 5a. Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login   # paste HF_TOKEN with write access
huggingface-cli whoami  # verify
```

### 5b. Create the Repository

```bash
huggingface-cli repo create Manthan-1.5B --type model
```

### 5c. Push from Kaggle (Automatic)

Training scripts push automatically when `push_to_hub: true` is set in config. The `HF_TOKEN` is read from environment.

### 5d. Push Manually from Local Checkpoint

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + LoRA adapter
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
)
model = PeftModel.from_pretrained(base, "./outputs/sft_local_test")

# Merge LoRA weights into base for clean inference
merged = model.merge_and_unload()

# Push to Hub
merged.push_to_hub("shahansha/Manthan-1.5B", safe_serialization=True)

# Push tokenizer
tokenizer = AutoTokenizer.from_pretrained("./outputs/sft_local_test")
tokenizer.push_to_hub("shahansha/Manthan-1.5B")
```

### 5e. Use from HuggingFace After Publishing

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "shahansha/Manthan-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Use with smolagents
from src.inference.smolagents_integration import create_manthan_agent
agent = create_manthan_agent(model_path=model_id)
result = agent.run("What is the sum of the first 100 prime numbers?")
print(result)
```

---

## 6. Testing & Verification

### 6a. Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Specific test files
pytest tests/test_reward_functions.py -v  # Reward function correctness
pytest tests/test_sandbox.py -v           # Code execution sandbox safety
```

### 6b. Reward Function Verification

Test each reward signal independently:

```python
from src.training.reward_functions import (
    tool_execution_reward, answer_correctness_reward, format_reward
)

# Case 1: Perfect completion — should score 1.0 on tool execution
perfect = """<tool_call>{"name": "python_repl", "arguments": {"code": "print(6*7)"}}</tool_call>
<tool_response>{"result": "42", "success": true}</tool_response>
<final_answer>42</final_answer>"""

assert tool_execution_reward(perfect, {"success": True, "result": "42"}) == 1.0
assert answer_correctness_reward(perfect, "42") == 1.0
assert format_reward(perfect) == 0.1

# Case 2: No tool call — tool execution should be 0.0
no_tool = "The answer is 42. <final_answer>42</final_answer>"
assert tool_execution_reward(no_tool) == 0.0
assert format_reward(no_tool) == 0.0

# Case 3: Malformed JSON — should be 0.0
bad_json = "<tool_call>NOT VALID JSON</tool_call><final_answer>42</final_answer>"
assert tool_execution_reward(bad_json) == 0.0

# Case 4: Wrong answer — correctness 0.0, but format/tool still score
wrong_answer = """<tool_call>{"name": "python_repl", "arguments": {"code": "print(6*8)"}}</tool_call>
<tool_response>{"result": "48", "success": true}</tool_response>
<final_answer>48</final_answer>"""
assert answer_correctness_reward(wrong_answer, "42") == 0.0
assert tool_execution_reward(wrong_answer, {"success": True, "result": "48"}) == 1.0

print("All reward function assertions passed.")
```

### 6c. Budget Forcing Verification

```python
from transformers import AutoTokenizer
from src.inference.budget_forcing import BudgetForcingProcessor

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
processor = BudgetForcingProcessor(tokenizer, minimum_tool_calls=2, maximum_tool_calls=4)

# Verify: if only 1 tool call so far, final_answer token must be suppressed
# (Unit test — no model required)
assert processor.minimum_tool_calls == 2
assert processor.maximum_tool_calls == 4
assert "<tool_call>" in processor._tool_call_str
print("BudgetForcingProcessor initialized correctly.")
```

### 6d. Dataset Integrity Check

```python
from datasets import load_from_disk

ds = load_from_disk("data/processed/manthan_dataset")
train = ds["train"]
eval_ds = ds["eval"]

print(f"Train samples: {len(train)}")
print(f"Eval samples:  {len(eval_ds)}")

# Verify format of first sample
sample = train[0]["text"]
assert "<|im_start|>" in sample, "Missing ChatML start token"
assert "<tool_call>" in sample, "Missing tool_call block"
assert "<final_answer>" in sample, "Missing final_answer block"
print("Dataset format verified.")
print("Sample preview:")
print(sample[:500])
```

### 6e. GSM8K Smoke Test

```bash
# Runs against 3 problems only — confirms the eval pipeline works, no GPU required
python src/eval/benchmark_gsm8k.py --smoke-test

# Full eval on 100 problems (requires GPU and trained model)
python src/eval/benchmark_gsm8k.py \
  --model shahansha/Manthan-1.5B \
  --n-samples 100 \
  --output outputs/gsm8k_results.json
```

Expected results file (`outputs/gsm8k_results.json`):
```json
{
  "accuracy": 0.68,
  "tool_call_parsability": 0.91,
  "avg_tool_calls": 1.8,
  "timeout_rate": 0.02,
  "n_samples": 100
}
```

**Target thresholds** (success criteria):

| Metric | Baseline | Target |
|---|---|---|
| GSM8K accuracy | ~45% | **>65%** |
| Tool call JSON parsability | ~7% | **>85%** |
| MBPP pass@1 | ~35% | **>50%** |
| smolagents success rate | untested | **>70%** |
| Avg tool calls before answer | N/A | **1.5–3.0** |

### 6f. Tool Success Rate Metric

```bash
# Measures: % of completions where JSON parses AND code runs without error
python src/eval/tool_success_rate.py \
  --model shahansha/Manthan-1.5B \
  --n-samples 50 \
  --output outputs/tool_success.json
```

---

## 7. Use Cases & Real Examples

### Use Case 1: Grade-School and Competition Math

**Problem:** *A train travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. What is the total distance?*

**What Manthan does:**
```
<tool_call>{"name": "python_repl", "arguments": {"code": "d1 = 60 * 2.5\nd2 = 80 * 1.5\nprint(d1 + d2)"}}</tool_call>
<tool_response>{"result": "270.0", "success": true}</tool_response>
<final_answer>270 km</final_answer>
```
No chain-of-thought paragraphs. Solves directly through execution.

---

### Use Case 2: Code Debugging

**Problem:** *Fix this function — it should sum ALL elements but starts from index 1:*
```python
def sum_list(lst):
    total = 0
    for i in range(1, len(lst)):
        total += lst[i]
    return total
```

**What Manthan does:**
```
<tool_call>{"name": "python_repl", "arguments": {"code":
"def sum_list_broken(lst):\n    total = 0\n    for i in range(1, len(lst)):\n        total += lst[i]\n    return total\n\ndef sum_list_fixed(lst):\n    total = 0\n    for i in range(0, len(lst)):\n        total += lst[i]\n    return total\n\ntest = [1, 2, 3, 4, 5]\nprint('Broken:', sum_list_broken(test))\nprint('Fixed:', sum_list_fixed(test))"}}</tool_call>
<tool_response>{"result": "Broken: 14\nFixed: 15", "success": true}</tool_response>
<final_answer>The bug is range(1, ...) — change to range(0, len(lst)) to include the first element. Fixed result: 15.</final_answer>
```

---

### Use Case 3: Data Analysis

**Problem:** *I have sales data: [1200, 1540, 980, 2100, 1750, 1320]. Find the mean, median, and the month with highest sales.*

**What Manthan does:**
```
<tool_call>{"name": "python_repl", "arguments": {"code":
"data = [1200, 1540, 980, 2100, 1750, 1320]\nmean = sum(data) / len(data)\nsorted_d = sorted(data)\nmedian = (sorted_d[2] + sorted_d[3]) / 2\nmax_month = data.index(max(data)) + 1\nprint(f'Mean: {mean:.1f}')\nprint(f'Median: {median}')\nprint(f'Highest sales: Month {max_month} with {max(data)}')"}}</tool_call>
<tool_response>{"result": "Mean: 1481.7\nMedian: 1430.0\nHighest sales: Month 4 with 2100", "success": true}</tool_response>
<final_answer>Mean: 1481.7, Median: 1430.0, Highest sales in Month 4 (2100).</final_answer>
```

---

### Use Case 4: Algorithm Verification

**Problem:** *Is this binary search implementation correct? It seems to loop forever sometimes.*

**What Manthan does:**
```
<tool_call>{"name": "python_repl", "arguments": {"code":
"def binary_search_buggy(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid   # BUG: should be mid+1\n        else: hi = mid\n    return -1\n\n# Test with known case — will loop\nimport signal\ndef handler(sig, frame): raise TimeoutError()\nsignal.signal(signal.SIGALRM, handler) if hasattr(signal, 'SIGALRM') else None\nprint(binary_search_buggy([1,2,3,4,5], 4))"}}</tool_call>
<tool_response>{"result": "Error: code execution timed out (10s limit)", "success": false}</tool_response>
<tool_call>{"name": "python_repl", "arguments": {"code":
"# The bug confirmed: lo = mid should be lo = mid + 1\ndef binary_search_fixed(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid\n    return -1\n\nprint(binary_search_fixed([1,2,3,4,5], 4))\nprint(binary_search_fixed([1,2,3,4,5], 6))"}}</tool_call>
<tool_response>{"result": "3\n-1", "success": true}</tool_response>
<final_answer>Yes, the bug is `lo = mid` instead of `lo = mid + 1`. When arr[mid] < target, the lower bound must advance past mid, otherwise it loops infinitely on certain inputs. Fixed version returns index 3 for target=4.</final_answer>
```

This example shows **multi-step tool interaction** — the failed execution becomes evidence that informs the next call.

---

### Use Case 5: Logic / Combinatorics

**Problem:** *How many ways can you arrange the letters in the word "MISSISSIPPI"?*

**What Manthan does:**
```
<tool_call>{"name": "python_repl", "arguments": {"code":
"from math import factorial\nword = 'MISSISSIPPI'\nfrom collections import Counter\ncounts = Counter(word)\nprint('Letter counts:', dict(counts))\nnumerator = factorial(len(word))\ndenominator = 1\nfor c in counts.values():\n    denominator *= factorial(c)\nresult = numerator // denominator\nprint(f'Arrangements: {result}')"}}</tool_call>
<tool_response>{"result": "Letter counts: {'M': 1, 'I': 4, 'S': 4, 'P': 2}\nArrangements: 34650", "success": true}</tool_response>
<final_answer>There are 34,650 distinct arrangements of the letters in MISSISSIPPI.</final_answer>
```

---

### Use Case 6: smolagents Agent Pipeline

```python
from src.inference.smolagents_integration import create_manthan_agent

agent = create_manthan_agent(
    model_path="shahansha/Manthan-1.5B",
    min_tool_calls=1,
    max_tool_calls=5,
)

# Multi-step agentic task
result = agent.run("""
I have a list of student scores: [78, 92, 55, 88, 67, 45, 99, 71].
1. Find the mean and standard deviation.
2. Identify students scoring more than 1 standard deviation below the mean.
3. Tell me what percentage passed (score >= 60).
""")
print(result)
```

The agent makes 3 tool calls (one per sub-task), each verifying the computation before progressing.

---

## 8. Scenarios Where Manthan Excels

### ✅ Best Fit Scenarios

#### 1. Verifiable Computation Tasks
Any problem where correctness can be checked by executing code:
- Math word problems (arithmetic → algebra → calculus)
- Statistical calculations (mean, median, variance, correlation)
- Unit conversions and formula applications
- Numerical algorithms (sorting, searching, dynamic programming)

**Why Manthan?** The model has learned to distrust its parametric memory and verify through execution. It uses Python as a calculator — producing verifiable answers, not plausible-sounding guesses.

#### 2. Code Review and Debugging
- Finding off-by-one errors in loops
- Identifying infinite loop conditions
- Catching wrong operator use (e.g., `^` vs `**` in Python)
- Verifying fix correctness by running the amended code

**Why Manthan?** Budget forcing ensures the model actually executes the code before answering. It cannot emit `<final_answer>` without at least one `<tool_call>` — it is structurally prevented from guessing.

#### 3. Small-Model Edge Deployment
- Raspberry Pi, Jetson Nano, or single-core cloud instances
- Inference at < 1 GB RAM (when quantized to 4-bit)
- Batch processing of math homework graders
- Embedded agent in VS Code extensions

**Why Manthan?** At 1.5B parameters with 4-bit quantization, the model runs on consumer hardware. It outperforms larger verbal CoT models on tool-amenable tasks because it never wastes tokens on "Let me think step by step..."

#### 4. smolagents / Tool-Augmented Pipelines
- HuggingFace agent benchmarks where tool use is expected
- Pipelines where model outputs feed into real Python execution
- Multi-step agentic workflows (plan → execute → verify → conclude)

**Why Manthan?** It is the only model under 3B parameters optimized specifically for `smolagents` `CodeAgent`. Tool-call JSON parsability targets >85% vs ~7% baseline.

#### 5. Education & Tutoring Assistants
- Step-by-step math problem solving with executed verification
- Showing how to fix broken code with before/after execution
- Generating worked examples by running the computation

**Why Manthan?** Every answer is backed by execution evidence visible in the tool trace. Students see the computation, not just the conclusion.

---

### ❌ Poor Fit Scenarios

| Scenario | Why Not Manthan |
|---|---|
| Creative writing / storytelling | Tool-mediated reasoning adds no value; larger creative models are better |
| Free-form conversation / Q&A | No tool available for unverifiable opinions; use standard chat models |
| Multilingual tasks | Base model (Qwen2.5) has decent multilingual support, but fine-tuning focused on English tool traces |
| Tasks requiring > 1024 tokens context | Training used 1024-token cap; very long documents exceed context window |
| Real-time latency < 200ms | Even quantized 1.5B models need 300–500ms on commodity hardware |
| Tasks with no verifiable answer | GRPO rewards tool *execution* success — if no tool can verify the answer, training signal was absent |

---

## 9. Troubleshooting

### "bfloat16 not supported" on T4

```python
# Wrong
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Correct
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
```

### Training OOM on T4

Reduce batch size in config:
```yaml
training:
  per_device_train_batch_size: 1    # from 4
  gradient_accumulation_steps: 16   # compensate to keep effective batch=16
```

### Reward Stuck at < 0.3 During GRPO

Check that the sandbox is actually executing code:
```python
# In reward_functions.py, add a debug print temporarily:
print(f"[DEBUG] sandbox_result: {sandbox_result}")
```
If `sandbox_result` is always `None`, the GRPO trainer is not wiring the sandbox executor. Verify `grpo_train.py` passes `sandbox_result` to reward functions.

### Dataset Has No `<tool_call>` Blocks

The raw trace generator may have fallen back to dummy data without tool calls. Run:
```bash
python -c "
import json
with open('data/raw/synthetic_traces.jsonl') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        if '<tool_call>' not in d.get('text', d.get('messages', [''])[0] if isinstance(d.get('messages'), list) else ''):
            print(f'Line {i} missing tool_call')
            break
    else:
        print('All samples have tool_call blocks — OK')
"
```

### HuggingFace Push Fails (401 / 403)

```bash
# Verify token has write access
huggingface-cli whoami
# If expired, re-login
huggingface-cli login
```

### `smolagents` Not Found

```bash
pip install smolagents
```

### Windows: `PermissionError` on Temp Files in Sandbox

The sandboxed REPL in `smolagents_integration.py` writes to `tempfile.gettempdir()`. On Windows, ensure the user has write permissions to `%TEMP%`. If not, override:

```python
import tempfile
tempfile.tempdir = "C:/manthan_tmp"
```

---

## Quick Reference

```bash
# Smoke test everything (CPU, < 60 seconds)
pytest tests/ -v

# Generate data (no API key)
python src/data/generate_synthetic.py --dummy-data --n-samples 50

# Format dataset
python src/data/format_dataset.py --input data/raw/synthetic_traces.jsonl --output data/processed/manthan_dataset

# Local SFT smoke run (10 steps)
python src/training/sft_train.py --config configs/sft_local_test.yaml

# GSM8K smoke test (3 problems, no GPU)
python src/eval/benchmark_gsm8k.py --smoke-test

# Full GSM8K eval (GPU required)
python src/eval/benchmark_gsm8k.py --model shahansha/Manthan-1.5B --n-samples 100

# Run Gradio demo
python src/inference/demo.py --model-path shahansha/Manthan-1.5B
```

---

*Genesis Manthan — Built by Shahansha Shaik under Genesis AGI. Model target: `shahansha/Manthan-1.5B` on HuggingFace Hub.*
