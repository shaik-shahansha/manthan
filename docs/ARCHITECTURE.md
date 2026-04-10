# Genesis Manthan — Architecture Reference

## Model Architecture

```
Base: Qwen2.5-1.5B-Instruct (Apache 2.0)
Quantization: 4-bit QLoRA via Unsloth
LoRA rank: 16, alpha: 16
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Context length: 1024 tokens
```

## Training Phases

### Phase 1: SFT (Supervised Fine-Tuning)
```
Input:  ChatML traces with tool_call + tool_response roles
Loss:   Standard cross-entropy on assistant tokens only
Data:   ~7K examples (2K synthetic + 5K curated)
Time:   ~2–3 hours on Kaggle T4
VRAM:   ~5–7 GB
```

### Phase 2: GRPO (Group Relative Policy Optimization)
```
Generator:  SFT checkpoint from Phase 1
Reward:     tool_execution × 0.5 + answer_correctness × 0.4 + format × 0.1
Sampling:   4 completions per prompt, group-relative advantage
Data:       ~500 problems with verifiable answers (GSM8K/MBPP/TriviaQA)
Time:       ~20–25 hours on Kaggle T4 (across 3 sessions)
VRAM:       ~7–9 GB
Min steps:  300 (reward improvement plateau)
```

### Phase 3: Budget Forcing (Inference-only)
```
Mechanism:  LogitsProcessor suppresses <final_answer> until min_tool_calls reached
No training required — zero GPU cost
Config:     min_tool_calls=1 (default), max_tool_calls=5 (default)
```

## Data Flow

```
Raw Sources
    ├── glaiveai/glaive-function-calling-v2  (113K → filter → 5K)
    ├── NousResearch/hermes-function-calling-v1  (filter → 2K)
    └── Synthetic API generation (100 seeds → 2K traces)
         │
         ▼
format_dataset.py
    ├── Validate format (tool_call + tool_response + final_answer)
    ├── Convert to ChatML <|im_start|>...<|im_end|> format
    ├── Filter samples exceeding 1024 tokens
    └── Split 90/10 train/eval
         │
         ├── SFT Dataset → sft_train.py → Manthan-1.5B-sft-v0.1
         └── GRPO Reward Dataset → grpo_train.py → Manthan-1.5B-grpo-v0.1
                                                           │
                                                   budget_forcing.py
                                                           │
                                                   Manthan-1.5B (final)
```

## Reward Signal Architecture

```
completion (string)
    │
    ├── tool_execution_reward()
    │     ├── Parse <tool_call> JSON block → +0.5 if valid
    │     └── Execute in sandbox subprocess → +0.5 if success
    │
    ├── answer_correctness_reward()
    │     ├── Extract <final_answer> content
    │     └── Compare to ground truth → 0.0–1.0 (numeric tolerance)
    │
    ├── format_reward()
    │     └── 0.1 if ≥1 tool_call present, else 0.0
    │
    └── combined_reward() → weighted sum → GRPO trainer
```

## Inference Architecture (smolagents)

```
User query
    │
    ▼
create_manthan_agent() → smolagents CodeAgent
    │
    ├── BudgetForcingProcessor(min=1, max=5)
    │
    ├── python_repl_tool (sandboxed subprocess)
    │
    └── model.generate() → <tool_call> → execute → <tool_response> → ...
                                                                      │
                                                              <final_answer>
```

## Hardware Constraints (Kaggle T4)

```
GPU:     NVIDIA T4
VRAM:    16 GB
Compute: 8.1 TFLOPS FP32
Dtype:   FP16 ONLY (no bfloat16)

Memory budget:
  Base model (4-bit):    ~1.5 GB
  LoRA adapters:         ~0.2 GB
  Optimizer states:      ~1.0 GB
  Activations (grad ckpt): ~0.5 GB
  GRPO generations (×4): ~3.0 GB
  Total estimate:        ~6.2 GB (well within 16 GB)
```

## Key Token IDs (Qwen2.5 tokenizer)

These are looked up dynamically in __init__ — never hardcoded:
- `<tool_call>` — looked up via tokenizer.encode("<tool_call>")
- `<final_answer>` — looked up via tokenizer.encode("<final_answer>")
- `Wait` — looked up via tokenizer.encode("Wait")
- Special tokens: `<|im_start|>`, `<|im_end|>` are part of Qwen2.5 vocabulary

## Research References

| Paper | arXiv | Relevance |
|---|---|---|
| Tool-mediated reasoning in small models | 2507.05065 | Foundational paradigm |
| Budget forcing / s1 paper | 2510.21398 | Phase 3 technique |
| GRPO / DeepSeek-R1 | 2501.12599 | Training algorithm |
| Unsloth GRPO blog | unsloth.ai/blog/grpo | T4-specific implementation |
