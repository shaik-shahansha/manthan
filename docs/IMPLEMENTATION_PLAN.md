# Genesis Manthan — Implementation Plan

> **Manthan-1.5B**: The first open SLM that reasons through tool interaction, not chain-of-thought.  
> **Total cost**: $0 | **Total GPU hours**: ~35 | **Timeline**: 6 weeks

---

## Phase Overview

```
Week 1–2: Environment Setup + Dataset Generation (0 GPU hours)
Week 2–3: Phase 1 — SFT with QLoRA                (~5 GPU hours)
Week 3–4: Phase 2 — GRPO with Tool-Execution Rewards (~25 GPU hours)
Week 5:   Phase 3 — Budget Forcing + Evaluation    (~5 GPU hours)
Week 6:   Publishing + Launch                      (0 GPU hours)
```

---

## Week 1: Environment + Dataset Foundation

### Goals
- Working local dev environment (VS Code, CPU-only)
- 100+ synthetic tool-interaction traces validated
- Project scaffold fully committed to GitHub

### Local Setup Steps

```bash
# 1. Python 3.10+ virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dev dependencies (CPU-only)
pip install -r requirements-dev.txt

# 3. Verify tokenizer loads
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); print('OK')"

# 4. Run all smoke tests
python src/data/generate_synthetic.py --smoke-test
python src/data/format_dataset.py --smoke-test
python src/training/reward_functions.py --smoke-test
python src/inference/budget_forcing.py --smoke-test
```

### Deliverables
- [ ] `src/data/generate_synthetic.py` — working with `--smoke-test`
- [ ] `src/data/format_dataset.py` — ChatML conversion verified
- [ ] `data/raw/sample_traces.jsonl` — 100 handcrafted traces
- [ ] `configs/sft_config.yaml` — SFT hyperparameters
- [ ] All tests passing: `pytest tests/ -x`

### Synthetic Data Generation

Use this prompt with Claude or GPT-4o-mini (free tier):

```
You are generating training data for a small language model that solves problems 
through tool interaction rather than verbal reasoning.

Generate a complete tool-interaction trace for this problem:
[INSERT GSM8K PROBLEM]

Format your response EXACTLY as:
<tool_call>{"name": "python_repl", "arguments": {"code": "[python code]"}}</tool_call>
<tool_response>{"result": "[execution output]", "success": true}</tool_response>
<tool_call>{"name": "python_repl", "arguments": {"code": "[follow-up code if needed]"}}</tool_call>
<tool_response>{"result": "[execution output]", "success": true}</tool_response>
<final_answer>[answer]</final_answer>

Rules:
- NO verbal chain-of-thought before tool calls
- Each tool call must have a corresponding tool_response
- The final_answer comes ONLY after tool execution confirms it
- Include 1-3 tool calls minimum
```

---

## Week 2: Data Pipeline + SFT Preparation

### Goals
- Complete dataset pipeline (raw traces → ChatML → HuggingFace Dataset)
- ~2K synthetic traces + 5K curated from glaive/hermes datasets
- SFT training script validated locally with `--smoke-test`

### Dataset Pipeline

```
data/
├── raw/
│   ├── synthetic_traces.jsonl        ← Generated via API
│   ├── glaive_filtered.jsonl         ← Filtered from glaive-function-calling-v2
│   └── hermes_filtered.jsonl         ← Filtered from hermes-function-calling-v1
├── processed/
│   ├── train_chatml.jsonl            ← ChatML formatted, ready for SFT
│   ├── eval_chatml.jsonl             ← Held-out evaluation set
│   └── reward_dataset.jsonl          ← GSM8K/MBPP for GRPO reward checking
└── stats/
    └── dataset_stats.json            ← Token counts, format validation
```

### ChatML Format Specification

Every training sample must follow this exact structure:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are Genesis Manthan, an AI agent that solves problems by calling tools. Never reason verbally — always reason through tool execution."
    },
    {
      "role": "user", 
      "content": "A store has 48 apples. They sell 3/4 of them. How many remain?"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"python_repl\", \"arguments\": {\"code\": \"apples = 48\\nsold = int(48 * 3/4)\\nremain = apples - sold\\nprint(remain)\"}}</tool_call>"
    },
    {
      "role": "tool",
      "content": "{\"result\": \"12\", \"success\": true}"
    },
    {
      "role": "assistant",
      "content": "<final_answer>12 apples remain.</final_answer>"
    }
  ]
}
```

### Dataset Quality Checklist
- [ ] Every sample has at least 1 `<tool_call>` before `<final_answer>`
- [ ] No sample has verbal chain-of-thought reasoning
- [ ] Tool responses always include `"success": true/false`
- [ ] Token count: 95th percentile < 1024 tokens (fit in context window)
- [ ] Train/eval split: 90/10

---

## Week 2–3: Phase 1 — Supervised Fine-Tuning (SFT)

### Goals
- SFT checkpoint uploaded to HuggingFace Hub as `shahansha/Manthan-1.5B-sft-v0.1`
- Baseline evaluation: GSM8K tool-augmented accuracy
- Tool call JSON parsability baseline measured

### Kaggle Notebook: `02_sft_kaggle.ipynb`

```python
# Cell 1: Install dependencies
!pip install unsloth[kaggle-new] trl>=0.9.0 transformers>=4.45.0

# Cell 2: Load model (4-bit QLoRA)
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=1024,
    dtype=torch.float16,  # CRITICAL: never bfloat16 on T4
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # CRITICAL for T4 memory
    random_state=42,
)
```

### SFT Hyperparameters

```yaml
# configs/sft_config.yaml
model:
  name: "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
  max_seq_length: 1024
  dtype: "float16"
  load_in_4bit: true

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  learning_rate: 2.0e-4
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  seed: 42
  output_dir: "./outputs/sft_v1"

hub:
  push_to_hub: true
  hub_model_id: "shahansha/Manthan-1.5B-sft-v0.1"
```

### Expected SFT Results
- Training time: ~2–3 hours on T4
- VRAM usage: ~5–7GB (well within 16GB T4 limit)
- Expected tool call parsability improvement: 7% → ~40–60%

---

## Week 3–4: Phase 2 — GRPO with Tool-Execution Rewards

### Goals
- GRPO-trained checkpoint: `shahansha/Manthan-1.5B-grpo-v0.1`
- Tool call JSON parsability: >85%
- GSM8K tool-augmented accuracy: >65%
- Average tool calls per problem: 1.5–3.0

### Reward Function Architecture

Three composable reward functions, each returns `float` in `[0.0, 1.0]`:

```python
# src/training/reward_functions.py

def tool_execution_reward(completion: str, sandbox) -> float:
    """
    0.5 if tool_call parses as valid JSON
    +0.5 if execution succeeds (sandbox returns success=True)
    """
    ...

def answer_correctness_reward(completion: str, ground_truth: str) -> float:
    """
    1.0 for exact match
    0.5–0.9 for fuzzy numeric match (within 1%)
    0.0 for wrong answer
    """
    ...

def format_reward(completion: str) -> float:
    """
    0.1 if output contains at least one <tool_call> block
    0.0 if no tool call present (penalizes verbal CoT)
    """
    ...
```

### GRPO Hyperparameters

```yaml
# configs/grpo_config.yaml
training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_generations: 4          # GRPO samples 4 completions per prompt
  max_new_tokens: 512
  learning_rate: 5.0e-6
  warmup_steps: 20
  min_steps: 300              # minimum steps needed for reward improvement
  seed: 42
  output_dir: "./outputs/grpo_v1"

rewards:
  tool_execution_weight: 0.5
  answer_correctness_weight: 0.4
  format_weight: 0.1

hub:
  push_to_hub: true
  hub_model_id: "shahansha/Manthan-1.5B-grpo-v0.1"
```

### Kaggle Session Strategy (30 hrs/week cap)

| Session | Duration | Steps | Checkpoint |
|---|---|---|---|
| Session 1 | ~8 hrs | 0–150 | grpo-ckpt-150 |
| Session 2 | ~8 hrs | 150–300 | grpo-ckpt-300 ← minimum viable |
| Session 3 | ~8 hrs | 300–450 | grpo-ckpt-450 ← target |

Push each checkpoint to HuggingFace Hub immediately after session ends (before timeout).

---

## Week 5: Phase 3 — Budget Forcing + Evaluation

### Budget Forcing Implementation

Budget forcing is implemented as a `LogitsProcessor` subclass:

```python
# src/inference/budget_forcing.py

class BudgetForcingProcessor(LogitsProcessor):
    """
    Forces additional tool calls before model is allowed to produce <final_answer>.
    
    Injects 'Wait' token when:
    - Model generates <final_answer> token
    - Before minimum_tool_calls threshold is reached
    
    Blocks generation when:
    - maximum_tool_calls has been reached (forces conclusion)
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        minimum_tool_calls: int = 1,
        maximum_tool_calls: int = 5,
    ):
        ...
```

### Evaluation Suite

Run all benchmarks with:
```bash
python src/eval/benchmark_gsm8k.py --model shahansha/Manthan-1.5B --n-samples 100
python src/eval/benchmark_mbpp.py --model shahansha/Manthan-1.5B --n-samples 100
python src/eval/tool_success_rate.py --model shahansha/Manthan-1.5B
```

### Evaluation Targets

| Metric | Baseline (Qwen2.5-1.5B) | SFT v0.1 | GRPO v0.1 | Target |
|---|---|---|---|---|
| Tool call parsability | ~7% | ~50% | ~80% | **>85%** |
| GSM8K (tool-augmented) | ~45% | ~55% | ~62% | **>65%** |
| MBPP pass@1 | ~35% | ~42% | ~48% | **>50%** |
| Avg tool calls/problem | N/A | 1.2 | 1.8 | **1.5–3.0** |

---

## Week 6: Publishing & Launch

### HuggingFace Publishing Checklist

- [ ] Upload `safetensors` checkpoint to `shahansha/Manthan-1.5B`
- [ ] Write model card (use HuggingFace annotated template)
- [ ] Export GGUF: Q4_K_M, Q5_K_M, Q8_0 → upload to `shahansha/Manthan-1.5B-GGUF`
- [ ] Publish dataset → `shahansha/manthan-tool-reasoning-v1`
- [ ] Launch Gradio Space → `shahansha/Manthan-Demo`
- [ ] Apply for Community GPU Grant in Space settings

### Model Card Metadata

```yaml
---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- tool-calling
- reasoning
- grpo
- smolagents
- agentic
- genesis-agi
pipeline_tag: text-generation
library_name: transformers
datasets:
- shahansha/manthan-tool-reasoning-v1
- glaiveai/glaive-function-calling-v2
metrics:
- gsm8k
- mbpp
---
```

### Launch Sequence

| Day | Action |
|---|---|
| **Monday** | Publish model + dataset on HuggingFace |
| **Monday** | Post on r/LocalLLaMA (tag: [Model], [Training]), r/MachineLearning |
| **Monday** | Post on X/Twitter — tag @huggingface, @unaborax, @_philschmid |
| **Wednesday** | Publish HuggingFace community blog post (2,000 words) |
| **Friday** | Post GGUF versions, update community post with download stats |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| VRAM OOM on GRPO | Medium | High | Reduce `num_generations` from 4 to 2; enable gradient checkpointing |
| Synthetic data quality too low | Low | High | Manual review of 100 samples before training; automated format validation |
| Kaggle session timeout mid-training | Medium | Medium | Push checkpoint to HuggingFace every 50 steps |
| Tool execution sandbox security | Low | Medium | Run in restricted subprocess with timeout; no network access |
| GRPO reward hacking (model games reward) | Low | Medium | Monitor format vs correctness reward ratio; add KL penalty if needed |

---

## Local Testing Protocol

Before uploading ANY notebook to Kaggle, run locally:

```bash
# 1. Lint
ruff check src/ --fix

# 2. Type check
mypy src/ --ignore-missing-imports

# 3. Unit tests (CPU, no GPU required)
pytest tests/ -v --timeout=30

# 4. Smoke tests for all key scripts
python src/data/generate_synthetic.py --smoke-test
python src/data/format_dataset.py --smoke-test
python src/training/sft_train.py --smoke-test
python src/training/grpo_train.py --smoke-test
python src/training/reward_functions.py --smoke-test
python src/inference/budget_forcing.py --smoke-test
```

---

*Genesis AGI — Shahansha Shaik — 2025*
