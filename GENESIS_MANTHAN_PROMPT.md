# Genesis Manthan — Complete GitHub Copilot Agent Build Prompt

Copy the entire prompt below and paste it into **GitHub Copilot Chat** (Agent mode, `@workspace`).  
This is the master "start here" prompt. Use the section prompts for individual tasks.

---

## MASTER PROMPT — Paste This First

```
You are helping me build **Genesis Manthan (Manthan-1.5B)** — the first open small language model 
that reasons through tool interaction instead of chain-of-thought. I am Shahansha Shaik, working 
under the Genesis AGI brand.

**Project context:**
- HuggingFace target: shahansha/Manthan-1.5B
- Base model: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
- Training pipeline: QLoRA + SFT → GRPO (tool-execution rewards) → Budget Forcing
- Hardware: Kaggle T4 GPU (16GB, FP16 only — NEVER bfloat16)
- Local dev: VS Code on Windows, CPU-only for testing, Python 3.10+

**Core paradigm:**
The model learns to reason by calling tools (Python REPL, calculator, search) and observing 
results — NOT by verbal chain-of-thought. A sample training trace looks like:

  User: What is 17 × 23?
  Assistant: <tool_call>{"name":"python_repl","arguments":{"code":"print(17*23)"}}</tool_call>
  Tool: {"result":"391","success":true}
  Assistant: <final_answer>391</final_answer>

**Architecture rules:**
1. All training scripts must accept --smoke-test flag (tokenizer only, CPU, <10 seconds)
2. Never use bfloat16 — always torch.float16
3. Read secrets via os.environ.get("HF_TOKEN") — never hardcode
4. Keep data/training/eval in separate modules — no monolithic scripts
5. Reward functions must return float in [0.0, 1.0] and be composable
6. Budget forcing is a LogitsProcessor subclass — never post-processing

**Project structure is in .github/copilot-instructions.md and docs/IMPLEMENTATION_PLAN.md.**

All code you generate should:
- Work locally with --smoke-test (no GPU required)
- Be ready to run on Kaggle T4 without modification (just remove --smoke-test)
- Follow the file responsibilities in copilot-instructions.md exactly
- Include type hints on function signatures
```

---

## TASK PROMPTS — Use These for Individual Work Sessions

### PROMPT 1: Scaffold the Project

```
@workspace 
Create the complete project scaffold for Genesis Manthan. Create these files with working 
placeholder implementations:

1. requirements.txt — Kaggle training dependencies (unsloth[kaggle-new], trl, transformers, etc.)
2. requirements-dev.txt — Local CPU-only dev dependencies (NO unsloth, NO torch GPU)
3. src/__init__.py, src/data/__init__.py, src/training/__init__.py, 
   src/inference/__init__.py, src/eval/__init__.py — empty init files
4. tests/__init__.py — empty

Each source file must be importable and have a main() function + --smoke-test argparse pattern.

For requirements-dev.txt: transformers>=4.45.0, datasets>=2.20.0, tokenizers, anthropic, 
openai, gradio>=4.0.0, evaluate, pytest, ruff, mypy

Do NOT install unsloth locally — it's GPU-only.
```

---

### PROMPT 2: Build the Synthetic Data Generator

```
@workspace
Create src/data/generate_synthetic.py for Genesis Manthan.

This script calls the Anthropic Claude API (or OpenAI as fallback) to generate 
tool-interaction traces for training. 

Requirements:
- Config must use a dataclass, not raw dicts
- Accept --smoke-test flag: validates API key env var exists, prints "smoke test OK", exits
- Accept --n-samples INT (default 100) and --output PATH
- Read API key from os.environ.get("ANTHROPIC_API_KEY") — never hardcode
- Generate traces using the tool-mediated reasoning format (tool_call + tool_response + final_answer)
- Each trace MUST have at least 1 tool call — reject and regenerate if not
- Output format: JSONL, one JSON object per line
- Each JSONL record: {"problem": "...", "trace": "...", "source": "synthetic", "domain": "math|code|factual"}
- Domains: math (GSM8K-style), code_debug, factual_qa, data_analysis
- Include 25 seed problems per domain (100 total seeds hardcoded in the script)

The system prompt to send to Claude:
"You are generating training data for a language model that reasons ONLY through tool execution. 
Generate a tool-interaction trace. Use EXACTLY this format with no verbal reasoning:
<tool_call>{"name": "python_repl", "arguments": {"code": "..."}}</tool_call>
<tool_response>{"result": "...", "success": true}</tool_response>
<final_answer>...</final_answer>
Include 1-3 tool calls. No CoT, no explanations, only tool calls and final answer."

Type hints on all functions. Docstrings on public functions.
```

---

### PROMPT 3: Build the Dataset Formatter

```
@workspace
Create src/data/format_dataset.py for Genesis Manthan.

This script converts raw JSONL traces into a HuggingFace Dataset in ChatML format, 
ready for Unsloth SFT training.

Requirements:
- Accept --smoke-test: loads 3 hardcoded sample traces, validates format, exits
- Accept --input PATH (JSONL), --output PATH (HuggingFace Dataset directory)
- Accept --max-tokens INT (default 1024) — filter out samples exceeding this
- Use Qwen2.5 tokenizer (Qwen/Qwen2.5-1.5B-Instruct) for token counting
- Output DatasetDict with train (90%) and eval (10%) splits
- Each record in dataset must have field "text" containing the full ChatML string

ChatML format to produce:
<|im_start|>system
You are Genesis Manthan, an AI agent that solves problems by calling tools. Never reason 
verbally — always reason through tool execution.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{tool_calls_and_responses}<|im_end|>
<|im_start|>assistant
{final_answer}<|im_end|>

Validation function must check:
1. Contains at least one <tool_call> JSON block
2. Contains at least one <tool_response> JSON block  
3. tool_call content is valid JSON
4. tool_response content is valid JSON with "success" field
5. Contains <final_answer> block

Print stats at end: total samples, filtered count, token distribution (p50, p90, p95, max).
Type hints required. Use pathlib.Path throughout.
```

---

### PROMPT 4: Build the Reward Functions

```
@workspace
Create src/training/reward_functions.py for Genesis Manthan.

Three reward functions for GRPO training, each returning float in [0.0, 1.0].
Must work on CPU (no GPU dependency). Include --smoke-test that runs all functions 
with hardcoded mock inputs and verifies outputs are in [0.0, 1.0].

Function 1: tool_execution_reward(completion: str, sandbox_result: dict | None) -> float
  - 0.0 if no <tool_call> block found
  - 0.5 if <tool_call> found and content is valid JSON
  - 1.0 if sandbox_result is not None and sandbox_result["success"] == True
  - Handles malformed JSON gracefully (returns 0.0, no exceptions)

Function 2: answer_correctness_reward(completion: str, ground_truth: str) -> float
  - Extract answer from <final_answer>...</final_answer> tags
  - 1.0 for exact string match (case-insensitive, stripped)
  - For numeric answers: 1.0 if within 0.1%, 0.5 if within 1%, else 0.0
  - 0.0 if no <final_answer> tag found
  - 0.0 if completion contains <final_answer> but ground_truth is empty

Function 3: format_reward(completion: str) -> float
  - 0.1 if at least one <tool_call> block present, else 0.0
  - This penalizes verbal CoT by rewarding the presence of tool calls

Function 4 (combiner): combined_reward(completion, ground_truth, sandbox_result, weights) -> float
  - weights: dict with keys "tool_execution", "answer_correctness", "format"
  - Default weights: {"tool_execution": 0.5, "answer_correctness": 0.4, "format": 0.1}
  - Returns weighted sum, clipped to [0.0, 1.0]

All functions must handle None inputs without raising exceptions.
Type hints on all 4 functions. Add dataclass for RewardWeights config.
```

---

### PROMPT 5: Build the SFT Training Script

```
@workspace
Create src/training/sft_train.py for Genesis Manthan.

This is the Phase 1 supervised fine-tuning script using Unsloth + TRL SFTTrainer.
Must work with --smoke-test on CPU (tokenizer only, no model load, no training).

Requirements:
- Use pydantic BaseModel for SFTConfig (not argparse for config — but still accept CLI args)
- --smoke-test: verify tokenizer loads from Qwen/Qwen2.5-1.5B-Instruct, print config, exit
- --config PATH to YAML config file (load configs/sft_config.yaml by default)
- --local-rank for distributed training compatibility
- Detect Kaggle: os.environ.get("KAGGLE_KERNEL_RUN_TYPE")

When NOT smoke-testing (Kaggle execution):
1. Load model with FastLanguageModel.from_pretrained (dtype=torch.float16, load_in_4bit=True)
2. Apply LoRA with FastLanguageModel.get_peft_model (use_gradient_checkpointing="unsloth")
3. Load dataset from configs path or HuggingFace Hub
4. Create SFTTrainer with the ChatML "text" field
5. Call trainer.train()
6. Save to output_dir
7. If push_to_hub=True, push to Hub (read token from os.environ.get("HF_TOKEN"))

Critical: never import unsloth when smoke_test=True (it requires CUDA).
Use: if not args.smoke_test: from unsloth import FastLanguageModel

Print training loss every 10 steps. Log VRAM usage after model load.
Type hints required.
```

---

### PROMPT 6: Build the GRPO Training Script

```
@workspace
Create src/training/grpo_train.py for Genesis Manthan.

Phase 2 GRPO training using TRL GRPOTrainer with tool-execution rewards.
This is the core innovation — rewards come from tool execution success, not just answer correctness.

Requirements:
- --smoke-test: load tokenizer from Qwen/Qwen2.5-1.5B-Instruct, run reward functions 
  with mock data, print "GRPO smoke test OK", exit
- --config PATH (default configs/grpo_config.yaml)
- --resume-from-checkpoint PATH (for resuming across Kaggle sessions)

The reward function passed to GRPOTrainer must:
1. Extract all <tool_call> blocks from each completion
2. Execute each Python code snippet in a sandboxed subprocess (10 second timeout)
3. Call tool_execution_reward(), answer_correctness_reward(), format_reward() from reward_functions.py
4. Return combined_reward() as the scalar reward for GRPO

Sandbox implementation:
- Use subprocess.run() with timeout=10, capture_output=True
- Never allow file system access outside /tmp
- Pass code via stdin to a Python subprocess — do NOT use eval() or exec() directly in main process
- If subprocess times out, sandbox_result = {"success": False, "result": "timeout", "error": "TimeoutError"}

Checkpoint saving:
- Save every 50 steps to output_dir/checkpoint-{step}
- Push to HuggingFace Hub every 50 steps if push_to_hub=True
- This is critical so Kaggle session timeouts don't lose progress

Critical: never import unsloth when smoke_test=True
Type hints required. Docstrings on all public functions.
```

---

### PROMPT 7: Build Budget Forcing

```
@workspace
Create src/inference/budget_forcing.py for Genesis Manthan.

Implement budget forcing as a LogitsProcessor subclass from the transformers library.
This forces the model to make more tool calls before concluding.

Requirements:
- --smoke-test: instantiate BudgetForcingProcessor, run forward() with mock logits tensor 
  (CPU), verify output shape matches input, print "budget forcing smoke test OK", exit

BudgetForcingProcessor(LogitsProcessor):
  __init__(self, tokenizer, minimum_tool_calls=1, maximum_tool_calls=5)
  
  __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    For each item in batch:
    1. Decode input_ids to string
    2. Count <tool_call> occurrences in decoded string
    3. Count <final_answer> occurrences in decoded string
    4. If model is about to generate <final_answer> token AND tool_call_count < minimum_tool_calls:
       - Set score for <final_answer> token to -inf (suppress it)
       - Boost score for "Wait" token (find via tokenizer.encode("Wait"))
    5. If tool_call_count >= maximum_tool_calls:
       - Boost <final_answer> token score (force conclusion)
    Return modified scores

Also create a helper function: generate_with_budget_forcing(model, tokenizer, problem, 
  min_calls=1, max_calls=5) -> str
  that wraps model.generate() with BudgetForcingProcessor in logits_processor list.

Type hints required. docstrings on class and public methods.
Note: TokenIDs for <tool_call>, <final_answer>, and "Wait" must be looked up from tokenizer 
in __init__ — never hardcode token IDs.
```

---

### PROMPT 8: Build the smolagents Integration

```
@workspace
Create src/inference/smolagents_integration.py for Genesis Manthan.

Wrap the trained Manthan model as a smolagents-compatible agent.

Requirements:
- --smoke-test: print smolagents import status, create mock agent config, exit
- Support both CodeAgent and ToolCallingAgent from smolagents
- Read model path from argument or env var MANTHAN_MODEL_PATH
- Accept budget forcing config (min_calls, max_calls) as parameters
- Export a factory function: create_manthan_agent(agent_type="code", model_path=None, 
  min_tool_calls=1, max_tool_calls=5) -> Agent

The agent should work as a drop-in for any smolagents CodeAgent usage:
  agent = create_manthan_agent()
  result = agent.run("Calculate the sum of first 100 prime numbers")

Include a simple tool: python_repl_tool that executes Python code in a sandboxed subprocess 
(same security model as grpo_train.py sandbox — timeout=10, restricted filesystem).

If smolagents is not installed, print helpful error: 
  "Install smolagents: pip install smolagents"

Type hints required.
```

---

### PROMPT 9: Build the Evaluation Scripts

```
@workspace
Create src/eval/benchmark_gsm8k.py for Genesis Manthan.

Evaluates Manthan-1.5B on GSM8K using tool-augmented (tool-mediated) generation.

Requirements:
- --smoke-test: load 3 GSM8K samples, run format validation, print stats, exit (no model load)
- --model PATH or HuggingFace model ID (default: shahansha/Manthan-1.5B)
- --n-samples INT (default: 100, use -1 for full dataset)
- --use-budget-forcing BOOL (default: True)
- --min-tool-calls INT (default: 1)
- --output PATH for results JSON

Evaluation logic:
1. Load GSM8K test split from HuggingFace datasets
2. For each problem, generate answer using generate_with_budget_forcing()
3. Extract <final_answer> content
4. Compare to GSM8K ground truth (numeric comparison, 0.1% tolerance)
5. Track: accuracy, tool_call_count_per_problem, json_parsability_rate, timeout_rate

Output JSON format:
{
  "model": "...",
  "n_samples": 100,
  "accuracy": 0.67,
  "tool_call_parsability": 0.91,
  "avg_tool_calls_per_problem": 1.8,
  "timeout_rate": 0.02,
  "samples": [{"problem": "...", "generated": "...", "correct": true, "tool_calls": 2}]
}

Print live progress with running accuracy every 10 samples.
Type hints required.
```

---

### PROMPT 10: Build Kaggle Notebooks

```
@workspace
Create the Kaggle training notebook notebooks/02_sft_kaggle.ipynb for Genesis Manthan.

This notebook runs Phase 1 SFT on a Kaggle T4 GPU. Design all cells to be idempotent.

Cell structure:
1. [markdown] "# Genesis Manthan — Phase 1: Supervised Fine-Tuning"
2. [code] Install dependencies: !pip install unsloth[kaggle-new] trl>=0.9.0 transformers>=4.45.0 datasets peft accelerate
3. [code] Import all libraries, print versions, check GPU: torch.cuda.get_device_name(0)
4. [code] Load secrets: HF_TOKEN = os.environ.get("HF_TOKEN") — assert it's not None
5. [code] Load model with FastLanguageModel (dtype=torch.float16, use_gradient_checkpointing="unsloth")
6. [code] Print VRAM usage: print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB")
7. [code] Load dataset from shahansha/manthan-tool-reasoning-v1 (or local upload)
8. [code] Create and run SFTTrainer — use configs matching configs/sft_config.yaml
9. [code] Save model and push to Hub: model.push_to_hub("shahansha/Manthan-1.5B-sft-v0.1", token=HF_TOKEN)
10. [markdown] "## Results" 
11. [code] Quick inference test: generate 1 sample and print

Critical requirements:
- Never use bfloat16 anywhere — always torch.float16
- Always use_gradient_checkpointing="unsloth"
- Each cell should print confirmation when it completes successfully
- Notebook cells must be re-runnable (idempotent)
```

---

### PROMPT 11: Local Dev Notebook

```
@workspace
Create notebooks/00_local_dev_setup.ipynb for Genesis Manthan.

This is the local VS Code development notebook for CPU-based testing (no GPU needed).
Purpose: validate all code before uploading to Kaggle.

Cell structure:
1. [markdown] "# Genesis Manthan — Local Dev Setup & Smoke Tests"
   Explains: "Run all cells to validate the codebase before uploading to Kaggle. 
   No GPU required — all tests run on CPU in smoke-test mode."

2. [code] Check Python version (must be 3.10+), print system info

3. [code] Install dev dependencies: 
   import subprocess; subprocess.run(["pip", "install", "-r", "requirements-dev.txt"])

4. [code] Verify all source modules import successfully (no GPU-dependent imports):
   import src.data.generate_synthetic
   import src.data.format_dataset
   import src.training.reward_functions
   import src.inference.budget_forcing
   print("All imports OK")

5. [code] Run smoke test for data generator (subprocess call with --smoke-test)

6. [code] Run smoke test for format_dataset (subprocess call with --smoke-test)

7. [code] Test reward functions with mock data:
   from src.training.reward_functions import tool_execution_reward, answer_correctness_reward, format_reward
   (hardcoded test cases with expected outputs)

8. [code] Test budget forcing with mock tensors (CPU):
   from src.inference.budget_forcing import BudgetForcingProcessor

9. [code] Tokenizer validation: load Qwen/Qwen2.5-1.5B-Instruct tokenizer, 
   test encoding a sample ChatML trace, verify token count < 1024

10. [markdown] "## All checks passed! Ready to upload to Kaggle."
    Checklist of what was validated.
```

---

### PROMPT 12: Generate 100 Seed Problems

```
@workspace
In src/data/generate_synthetic.py, implement the SEED_PROBLEMS constant — 
a list of 100 seed problems for synthetic trace generation, 25 per domain:

Domain "math": GSM8K-style word problems involving arithmetic, percentages, rates.
Example: "A store had 240 items. They sold 35% on Monday and 25% of the remainder on Tuesday. How many items are left?"

Domain "code_debug": Short Python snippets with a bug to identify and fix.
Example: "This Python function should return the factorial of n but has a bug. Find and fix it: def factorial(n): if n == 0: return 1; return n * factorial(n-1) if n > 0 else factorial(-n)"

Domain "factual_qa": Questions answerable via Python computation or known formulas.
Example: "How many seconds are in a leap year?"

Domain "data_analysis": Problems involving lists, statistics, or data transformation.
Example: "Given the list [4, 7, 2, 9, 1, 5, 8, 3, 6, 10], find the median and mean, then return all values above the mean."

All 100 problems must be deterministic (same output every run).
Format: list of dicts: [{"domain": "math", "problem": "..."}, ...]
No external data downloads — all seed problems are hardcoded strings.
```

---

## DEBUGGING PROMPTS

### When VRAM OOM on Kaggle

```
@workspace
The GRPO training is running out of VRAM on Kaggle T4 (16GB). 
Looking at src/training/grpo_train.py, reduce memory usage without changing the algorithm:
1. Reduce num_generations from 4 to 2
2. Verify use_gradient_checkpointing="unsloth" is set
3. Add torch.cuda.empty_cache() calls between generations
4. Ensure dtype is torch.float16 everywhere (search for any bfloat16)
5. Reduce max_new_tokens in GRPO config from 512 to 256 if needed
Print VRAM usage before and after each change.
```

### When Reward Hacking Detected

```
@workspace
The GRPO training in src/training/grpo_train.py shows reward hacking — the model 
is generating <tool_call> tokens but with nonsense content (still getting format_reward=0.1).
Add a quality gate to tool_execution_reward in reward_functions.py:
- If tool_call JSON parses but "code" field is empty string or < 5 characters, return 0.0
- If tool_call executes but produces no output (empty result), return 0.25 instead of 1.0
Also add monitoring: log the ratio of format_reward / total_reward every 25 steps.
If this ratio exceeds 0.3, print a warning.
```

### When Kaggle Notebook Times Out

```
@workspace
The Kaggle GRPO training session timed out and I need to resume from checkpoint.
Update src/training/grpo_train.py to:
1. Accept --resume-from-checkpoint PATH argument
2. On resume: load the checkpoint, print which step it's resuming from
3. Add checkpoint saving every 25 steps (reduce from 50)
4. After each checkpoint save, push immediately to HuggingFace Hub with tag "checkpoint-{step}"
5. Add at training start: print("Estimated hours remaining: {:.1f}".format(remaining_steps * seconds_per_step / 3600))
```

---

## TESTING CHECKLIST (Run Before Every Kaggle Upload)

```
@workspace
Run the complete pre-Kaggle validation checklist for Genesis Manthan:

1. Lint: ruff check src/ --fix
2. Type check: mypy src/ --ignore-missing-imports  
3. Unit tests: pytest tests/ -v
4. Smoke tests (all must exit 0 in <10 seconds each):
   python src/data/generate_synthetic.py --smoke-test
   python src/data/format_dataset.py --smoke-test
   python src/training/reward_functions.py --smoke-test
   python src/training/sft_train.py --smoke-test
   python src/training/grpo_train.py --smoke-test
   python src/inference/budget_forcing.py --smoke-test

Report: which tests passed, which failed, and specific error messages for failures.
Do NOT proceed to Kaggle if any smoke test fails.
```

---

*Genesis AGI — Shahansha Shaik*  
*Paste into GitHub Copilot Chat (Agent mode) to begin any phase of Manthan development.*
