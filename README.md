# Genesis Manthan — मंथन-1.5B

> *"Small models shouldn't think in words — they should think through actions."*

**Genesis Manthan** is a small language model project focused on reasoning through **tool interaction** instead of chain-of-thought. It targets the Hugging Face and smolagents ecosystem, follows the tool-mediated reasoning setup described by Rainone et al. (arXiv:2507.05065), and is designed to be trainable on Kaggle T4 GPUs.

---

## What is Genesis Manthan?

| Property | Value |
|---|---|
| **Project name** | Genesis Manthan (मंथन = "churning of deep thought") |
| **Model name** | `Shahansha/Manthan-1.5B` |
| **Base model** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| **Training** | QLoRA + GRPO (tool-execution rewards) |
| **Unique paradigm** | Tool-mediated reasoning (not verbal chain-of-thought) |
| **Target platform** | HuggingFace + smolagents |
| **Total GPU cost** | ~30–40 hours (fits within Kaggle free tier) |
| **Total money cost** | $0 |

---

## Why It Matters

- **92.48%** of all HuggingFace downloads are models under 1B params — yet tool-calling sub-2B models barely exist
- HuggingFace's smolagents only references **7B–72B** models; zero optimized models exist under 3B
- Zero-shot function calling in sub-1B models achieves just **~7% JSON parsability**
- The SLM market grows from **$0.93B → $5.45B by 2032** (28.7% CAGR)
- The foundational paper (Rainone et al., 2025) proved tool-mediated reasoning outperforms verbal CoT in small models — but released **no models, no code**

---

## Core Ideas

1. Tool-mediated reasoning instead of verbal chain-of-thought
2. GRPO with tool-execution rewards, not only answer-match rewards
3. Budget forcing for agentic reasoning over tool-interaction traces
4. Small-model deployment targeting local workflows, Hugging Face, and smolagents

---

## Project Structure

```
manthan/
├── .github/
│   └── copilot-instructions.md       ← Copilot agent context
├── docs/
│   ├── IMPLEMENTATION_PLAN.md        ← Full 6-week plan
│   └── ARCHITECTURE.md               ← Model & training architecture
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py     ← Synthetic trace generation
│   │   ├── format_dataset.py         ← ChatML formatting
│   │   └── reward_dataset.py         ← GRPO reward dataset curation
│   ├── training/
│   │   ├── sft_train.py              ← Phase 1: Supervised fine-tuning
│   │   ├── grpo_train.py             ← Phase 2: GRPO training
│   │   └── reward_functions.py       ← Tool-execution reward signals
│   ├── inference/
│   │   ├── budget_forcing.py         ← Phase 3: Budget forcing
│   │   ├── smolagents_integration.py ← smolagents CodeAgent wrapper
│   │   └── demo.py                   ← Gradio demo script
│   └── eval/
│       ├── benchmark_gsm8k.py        ← GSM8K evaluation
│       ├── benchmark_mbpp.py         ← MBPP evaluation
│       └── tool_success_rate.py      ← Tool-execution success metric
├── notebooks/
│   ├── 00_local_dev_setup.ipynb      ← Local VS Code dev & smoke tests
│   ├── 02_sft_kaggle.ipynb           ← Kaggle SFT notebook
│   └── 03_grpo_kaggle.ipynb          ← Kaggle GRPO notebook
├── configs/
│   ├── sft_config.yaml               ← SFT hyperparameters
│   └── grpo_config.yaml              ← GRPO hyperparameters
├── GENESIS_MANTHAN_PROMPT.md         ← Complete Copilot agent build prompt
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quick Start (Local Dev)

```bash
# 1. Clone and enter repo
git clone https://github.com/shaik-shahansha/manthan.git
cd manthan

# 2. Create virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dev dependencies (CPU-only for local scaffolding)
pip install -r requirements-dev.txt

# 4. Run smoke test (loads tokenizer, no GPU needed)
python src/inference/budget_forcing.py --smoke-test

# 5. Smoke test the Gradio app
python src/inference/demo.py --smoke-test

# 6. Smoke test the interactive Hugging Face chat CLI
python src/inference/hf_chat.py --smoke-test

# 7. Open local dev notebook
# Open notebooks/00_local_dev_setup.ipynb in VS Code
```

## Hugging Face Release

- Model: [Shahansha/Manthan-1.5B](https://huggingface.co/Shahansha/Manthan-1.5B)
- Dataset: [Shahansha/manthan-tool-reasoning-v1](https://huggingface.co/datasets/Shahansha/manthan-tool-reasoning-v1)
- Deployment guide: [docs/HUGGINGFACE_DEPLOY.md](docs/HUGGINGFACE_DEPLOY.md)

The published model repo currently needs a full README body in the Hub model card. Use the production-ready template in [docs/HUGGINGFACE_DEPLOY.md](docs/HUGGINGFACE_DEPLOY.md) and push it to the model repo as `README.md`.

For a local UI check before touching the Space:

```bash
# Fast UI-only check
python src/inference/demo.py --demo-mode

# Interactive terminal chat with the published Hub model
python src/inference/hf_chat.py

# Run the actual Hub model locally
python src/inference/demo.py --model Shahansha/Manthan-1.5B

# Optional side-by-side comparison with the base model
python src/inference/demo.py --model Shahansha/Manthan-1.5B --compare-baseline
```

## Kaggle Training

1. Upload `notebooks/02_sft_kaggle.ipynb` to Kaggle
2. Enable **GPU T4 × 1** accelerator
3. Set Kaggle Secrets: `HF_TOKEN`, `WANDB_API_KEY` (optional)
4. Run all cells — ~3 hours for SFT, ~12–20 hours for GRPO (across sessions)

---

## Genesis AGI Connection

Genesis Manthan is the first model artifact under the **Genesis AGI** philosophy: agents that reason through *actions in the world* rather than internal monologue. Tool-mediated reasoning maps directly to Genesis AGI's "Digital Beings" concept — entities that perceive, act, observe, and iterate.

Future extensions:
- **Manthan-IoT**: Fine-tuned on ESP32 sensor API tool calls for edge intelligence
- **Manthan-Indic**: Hindi/Telugu tool-interaction traces for Indian construction workflows  
- **Manthan-2B+**: Scaled with curriculum RL on longer multi-step tool chains

---

## Citation

```bibtex
@misc{shaik2026manthan,
  title={Genesis Manthan-1.5B: Tool-Mediated Reasoning for Small Language Models},
  author={Shahansha Shaik},
  year={2026},
  url={https://huggingface.co/shahansha/Manthan-1.5B}
}
```

---

*Built by Shahansha Shaik — Genesis AGI*
