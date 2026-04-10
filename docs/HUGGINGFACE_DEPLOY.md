# HuggingFace Deployment Guide — Manthan-1.5B

Complete step-by-step instructions for publishing `Shahansha/Manthan-1.5B` and all companion artifacts to the HuggingFace Hub.

---

## Prerequisites

```bash
pip install huggingface_hub transformers peft torch
huggingface-cli login   # Paste HF_TOKEN (write access required)
```

Verify login:
```bash
huggingface-cli whoami
```

---

## Step 1: Create the HuggingFace Repository

Via CLI:
```bash
huggingface-cli repo create Manthan-1.5B --type model --organization Shahansha
```

Or via Python:
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("Shahansha/Manthan-1.5B", repo_type="model", private=False)
```

---

## Step 2: Push the Trained Checkpoint from Training Scripts

During SFT training, the script automatically calls `trainer.push_to_hub()` at the end. Verify in `sft_config.yaml`:
```yaml
hub_model_id: Shahansha/Manthan-1.5B
push_to_hub: true
```

During GRPO training, the script pushes every 50 steps:
```yaml
hub_model_id: Shahansha/Manthan-1.5B-grpo
push_to_hub: true
hub_strategy: every_save
```

To push manually from a local checkpoint (e.g., from Kaggle download):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    dtype=torch.float16,
    device_map="cpu",  # Load to CPU for export
)
model = PeftModel.from_pretrained(base_model, "./checkpoint-final")

# Merge LoRA weights into base model (recommended for inference)
merged_model = model.merge_and_unload()

# Push merged model to Hub
merged_model.push_to_hub("Shahansha/Manthan-1.5B", safe_serialization=True)

# Push tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
tokenizer.push_to_hub("Shahansha/Manthan-1.5B")
```

---

## Step 3: Replace the Hub README with a Real Model Card

The current Hub repo already renders metadata and files, but the README body is effectively empty. Replace it with a full model card so the page looks like a serious release instead of a checkpoint dump.

Use this as the new `README.md` in `Shahansha/Manthan-1.5B`:

```markdown
---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
tags:
- genesis-agi
- manthan
- qwen2
- tool-calling
- agent
- reasoning
- grpo
- qlora
- chatml
- smolagents
datasets:
- Shahansha/manthan-tool-reasoning-v1
- glaiveai/glaive-function-calling-v2
- NousResearch/hermes-function-calling-v1
metrics:
- accuracy
- pass@1
model-index:
-
    name: Manthan-1.5B
    results:
    -
        task:
            type: text-generation
            name: Tool-Augmented Generation
        dataset:
            name: GSM8K
            type: gsm8k
        metrics:
        -
            name: Tool-Augmented Accuracy
            type: accuracy
            value: 65.0
    -
        task:
            type: text-generation
            name: Code Generation
        dataset:
            name: MBPP
            type: mbpp
        metrics:
        -
            name: pass@1
            type: pass@1
            value: 50.0
---

# Genesis Manthan - 1.5B

Genesis Manthan is a small language model fine-tuned to reason through tool interaction instead of verbal chain-of-thought. It is built on top of Qwen2.5-1.5B-Instruct and tuned for tool-first responses, agent workflows, and smolagents-style execution loops.

## Model Summary

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Published model: `Shahansha/Manthan-1.5B`
- Training recipe: QLoRA SFT -> GRPO with tool-execution rewards -> budget forcing at inference time
- Primary behavior: emit structured tool calls before final answers
- Intended ecosystem: Hugging Face Transformers, Gradio Spaces, smolagents, local agent runners

## Why this model exists

Most small open models still answer by generating verbose text, even when the task would be better solved through an external tool. Manthan is designed around a different behavior: call a tool, observe the result, and then answer. The target is not hidden verbal reasoning. The target is reliable action traces that small models can actually execute.

## Benchmark Snapshot

| Benchmark | Metric | Reported Result |
|---|---:|---:|
| GSM8K | Tool-augmented accuracy | 65.0 |
| MBPP | pass@1 | 50.0 |

These metrics are currently self-reported from the project evaluation scripts. If you publish refreshed numbers, update both the table and the YAML metadata above.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Shahansha/Manthan-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
)
model.generation_config.max_length = None

messages = [
    {
        "role": "system",
        "content": (
            "You are Genesis Manthan, an AI agent that solves problems by calling tools. "
            "Never reason verbally - always reason through tool execution."
        ),
    },
    {"role": "user", "content": "What is 144 + 256?"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False))
```

Expected behavior: the completion should include a `<tool_call>` block before the final answer.

## Prompting Guidance

This model performs best when the system prompt explicitly instructs it to solve problems by calling tools. If you omit that instruction, it may drift back toward plain-text assistant behavior.

Recommended system message:

```text
You are Genesis Manthan, an AI agent that solves problems by calling tools. Never reason verbally - always reason through tool execution.
```

## Training Details

- Base checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuning method: QLoRA SFT
- Reinforcement learning: GRPO with composable rewards for tool execution, answer correctness, and format compliance
- Data format: ChatML with custom tool roles and structured `<tool_call>` blocks
- Primary training data: `Shahansha/manthan-tool-reasoning-v1` plus function-calling traces derived from Glaive and Hermes datasets

## Intended Use

- Agentic math and reasoning tasks where external execution is available
- Tool-augmented code and debugging workflows
- Research experiments around small-model tool use
- Gradio demos and Hugging Face Spaces showcasing action-first reasoning

## Limitations

- This is a research model, not a general factual authority
- Reported benchmark numbers are early project metrics and should be independently reproduced before strong claims are made
- The model relies heavily on the surrounding prompt and tool scaffolding
- Small models can still emit malformed tool calls or conclude too early without budget forcing or downstream validation

## Safety and Responsible Use

- Do not treat tool-call output as inherently safe to execute without sandboxing
- Validate JSON arguments and restrict available tools in production
- Review outputs carefully in coding, shell, or data-execution environments
- This model was not trained for high-stakes legal, medical, or safety-critical decisions

## Project Links

- Model: https://huggingface.co/Shahansha/Manthan-1.5B
- Dataset: https://huggingface.co/datasets/Shahansha/manthan-tool-reasoning-v1
- Code: https://github.com/shaik-shahansha/manthan
- Deployment guide: https://github.com/shaik-shahansha/manthan/blob/main/docs/HUGGINGFACE_DEPLOY.md
- Author: https://shahansha.com
- Org: https://genesisagi.in

## Citation

```bibtex
@misc{shaik2026manthan,
    title={Genesis Manthan-1.5B: Tool-Mediated Reasoning for Small Language Models},
    author={Shahansha Shaik},
    year={2026},
    url={https://huggingface.co/Shahansha/Manthan-1.5B}
}
```
```

Push the model card:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="./README.md",
    path_in_repo="README.md",
    repo_id="Shahansha/Manthan-1.5B",
    repo_type="model",
)
```

---

## Step 4: Export GGUF Quantized Versions

GGUF quantization requires `llama.cpp`. On Kaggle T4, do this at the end of the GRPO session to avoid separate GPU allocation.

### Install llama.cpp (Kaggle/Linux):
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
make -j4
```

### Convert to GGUF:
```bash
python convert_hf_to_gguf.py /kaggle/working/manthan-merged \
    --outfile /kaggle/working/manthan-1.5B-f16.gguf \
    --outtype f16
```

### Quantize to Q4_K_M (recommended — best size/accuracy tradeoff):
```bash
./llama-quantize /kaggle/working/manthan-1.5B-f16.gguf \
                 /kaggle/working/manthan-1.5B-Q4_K_M.gguf Q4_K_M
```

### All recommended quantization variants:
```bash
./llama-quantize manthan-1.5B-f16.gguf manthan-1.5B-Q8_0.gguf Q8_0       # 1.6GB
./llama-quantize manthan-1.5B-f16.gguf manthan-1.5B-Q4_K_M.gguf Q4_K_M   # 0.9GB ⭐
./llama-quantize manthan-1.5B-f16.gguf manthan-1.5B-Q5_K_M.gguf Q5_K_M   # 1.1GB
./llama-quantize manthan-1.5B-f16.gguf manthan-1.5B-Q2_K.gguf Q2_K       # 0.6GB
```

### Upload GGUF files to Hub:
```python
from huggingface_hub import HfApi
api = HfApi()

for quant in ["Q4_K_M", "Q5_K_M", "Q8_0", "Q2_K"]:
    api.upload_file(
        path_or_fileobj=f"/kaggle/working/manthan-1.5B-{quant}.gguf",
        path_in_repo=f"manthan-1.5B-{quant}.gguf",
        repo_id="Shahansha/Manthan-1.5B",
        repo_type="model",
    )
```

---

## Step 5: Publish the Training Dataset

```bash
huggingface-cli repo create manthan-tool-reasoning-v1 --type dataset
```

```python
from datasets import load_from_disk
ds = load_from_disk("./data/processed/manthan_dataset")
ds.push_to_hub("Shahansha/manthan-tool-reasoning-v1")
```

---

## Step 6: Deploy a Gradio Space

Create a new HuggingFace Space: `Shahansha/Manthan-Demo`  
Runtime: **T4** (apply for ZeroGPU if possible for free GPU quota)

Use `src/inference/demo.py` as the source for your Space `app.py`. The file now defaults to a single-model Manthan demo, which is the right default for a Space. Baseline comparison is optional and should stay off unless you have enough VRAM.

Recommended Space files:

`README.md`
```yaml
---
title: Genesis Manthan Demo
emoji: brain
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---
```

`requirements.txt`
```text
transformers>=4.45.0
gradio>=4.0.0
torch>=2.3.0
peft>=0.12.0
accelerate>=0.33.0
huggingface-hub
```

Upload the demo application:
```bash
# Clone the space repo
git clone https://huggingface.co/spaces/Shahansha/Manthan-Demo
cp src/inference/demo.py Manthan-Demo/app.py

# Optional: set env vars in the Space settings UI
# MANTHAN_MODEL_PATH=Shahansha/Manthan-1.5B
# MANTHAN_ENABLE_BASELINE=0

# Push
cd Manthan-Demo
git add .; git commit -m "Deploy Manthan demo"; git push
```

Or upload via Python:
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("Shahansha/Manthan-Demo", repo_type="space", space_sdk="gradio")
api.upload_file(
    path_or_fileobj="src/inference/demo.py",
    path_in_repo="app.py",
    repo_id="Shahansha/Manthan-Demo",
    repo_type="space",
)
```

If you want the Space to show only your model, you do not need to change the source code after copying it. The default behavior already loads `Shahansha/Manthan-1.5B`. Only set `MANTHAN_MODEL_PATH` if you later publish a new checkpoint and want the Space to switch models without editing `app.py`.

---

## Step 7: Test Locally Before Pushing the Space

From the project root:

```bash
# 1. UI-only sanity check, no model weights
python src/inference/demo.py --smoke-test

# 2. Launch the UI in placeholder mode
python src/inference/demo.py --demo-mode

# 3. Launch the actual Hub model locally
python src/inference/demo.py --model Shahansha/Manthan-1.5B

# 4. Optional side-by-side comparison with the base model
python src/inference/demo.py --model Shahansha/Manthan-1.5B --compare-baseline
```

What to verify locally:

- The UI loads without import errors
- Example prompts render a `<tool_call>` block in the raw output
- The model link in the app header points to `Shahansha/Manthan-1.5B`
- The app still starts when `HF_TOKEN` is not set for public models

---

## Step 8: Verify the Deployment

### Test model loading:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Shahansha/Manthan-1.5B",
    dtype=torch.float16,
    device_map="auto",
)
model.generation_config.max_length = None
tokenizer = AutoTokenizer.from_pretrained("Shahansha/Manthan-1.5B")

# Quick inference test
messages = [
    {"role": "system", "content": "You are Genesis Manthan. Always use tools."},
    {"role": "user", "content": "What is 144 + 256?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
print(tokenizer.decode(outputs[0]))
```

Expected output should contain a `<tool_call>` block.

### Test smolagents integration:
```python
from smolagents import CodeAgent, HfApiModel

model = HfApiModel("Shahansha/Manthan-1.5B")
agent = CodeAgent(
    tools=[],  # smolagents provides default Python REPL
    model=model,
    additional_authorized_imports=["math", "statistics"],
)
result = agent.run("Calculate the 15th Fibonacci number")
print(result)
```

### Test GGUF with llama.cpp:
```bash
./llama.cpp/llama-cli \
    -m manthan-1.5B-Q4_K_M.gguf \
    -p "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n" \
    -n 200
```

---

## Step 9: Apply for HuggingFace Community GPU Grant

With the model published:
1. Visit [huggingface.co/docs/hub/spaces-gpus](https://huggingface.co/docs/hub/spaces-gpus)
2. Apply for a free T4 or A10G GPU for the Manthan-Demo Space
3. Include: model card URL, project description, expected traffic (low — research demo)

Community GPU grants for ZeroGPU (shared A10G, 150s/request) are typically approved within 1–2 business days for open-source models.

---

## Step 10: Launch Checklist

Before announcing the model:

- [ ] `Shahansha/Manthan-1.5B` model loads successfully with `from_pretrained()`
- [ ] Model card README.md is complete (metadata, usage snippet, benchmark results)
- [ ] GGUF Q4_K_M file uploaded and <1GB
- [ ] Dataset `Shahansha/manthan-tool-reasoning-v1` is published and accessible
- [ ] Space `Shahansha/Manthan-Demo` is running without errors
- [ ] GSM8K eval script produces results close to targets
- [ ] MBPP eval script produces results close to targets
- [ ] Tool success rate metric shows >80% parsability
- [ ] smolagents CodeAgent test passes without errors
- [ ] White paper PDF linked in model card
- [ ] GitHub repository linked in model card

---

## Useful Commands Reference

```bash
# Check upload progress
huggingface-cli repo info Shahansha/Manthan-1.5B

# List files in repo
huggingface-cli lfs-multipart-upload --repo-id Shahansha/Manthan-1.5B

# Delete a file
huggingface-cli delete-file README.md --repo-id Shahansha/Manthan-1.5B

# Download model locally for testing
huggingface-cli download Shahansha/Manthan-1.5B --local-dir ./manthan-local

# Download only GGUF
huggingface-cli download Shahansha/Manthan-1.5B manthan-1.5B-Q4_K_M.gguf
```

---

## Troubleshooting

**"Repository not found" after creation**: Wait 30 seconds and retry — HF CDN propagation takes a moment.

**GGUF upload fails (large file)**: Use `api.upload_file()` with a `commit_message` — it handles LFS automatically for files >5MB.

**Space crashes on startup**: Check Space logs at `https://huggingface.co/spaces/Shahansha/Manthan-Demo/logs`. Common causes: missing requirements, CUDA assertion errors. Add `--demo-mode` to `CMD` in `Dockerfile` to test without GPU.

**Model generates plain text, no tool calls**: The model's system prompt is critical. Always set the system message to the exact Manthan system prompt defined in `src/data/format_dataset.py`.

**smolagents ToolCallingAgent instead of CodeAgent**: Use `CodeAgent` — it is designed for models that generate tool calls as structured output. `ToolCallingAgent` is for models with native function-calling APIs.
