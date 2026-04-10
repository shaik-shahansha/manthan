"""
Phase 2 GRPO training for Genesis Manthan.
Uses TRL GRPOTrainer with tool-execution rewards as the primary signal.

Run locally (GPU):   python src/training/grpo_train.py --config configs/grpo_config.yaml
Smoke test (CPU):    python src/training/grpo_train.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# Load .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass

# Ensure project root is on sys.path so `src.*` imports work when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─── config ───────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("grpo", cfg)


# ─── sandboxed code execution ─────────────────────────────────────────────────

def execute_code_sandbox(code: str, timeout_seconds: int = 10) -> dict:
    """
    Execute Python code in a restricted subprocess sandbox.

    Security model:
    - Code runs in a child process, not eval/exec in main process
    - 10-second hard timeout
    - Output capped at 10KB
    - No network access (relies on OS defaults; can be hardened with seccomp on Linux)

    Returns dict with keys: success (bool), result (str), error (str)
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir=tempfile.gettempdir(),
        prefix="manthan_sandbox_",
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=tempfile.gettempdir(),
        )
        stdout = result.stdout[:10240]  # cap at 10KB
        stderr = result.stderr[:2048]
        if result.returncode == 0:
            return {"success": True, "result": stdout.strip(), "error": ""}
        else:
            return {"success": False, "result": stdout.strip(), "error": stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "result": "", "error": "TimeoutError: code exceeded time limit"}
    except Exception as exc:
        return {"success": False, "result": "", "error": str(exc)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|<tool_response>|<final_answer>|$)", re.DOTALL)


def _completion_to_text(completion: object) -> str:
    """Normalise GRPO completions into a single text string."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _prompt_to_chatml(prompt: object) -> str:
    """Convert a prompt payload into the ChatML string used during SFT."""
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)

    return str(prompt)


def _extract_first_tool_call_payload(completion_text: str) -> str | None:
    """Return the raw JSON payload from the first tool call block."""
    match = _TOOL_CALL_RE.search(completion_text)
    if not match:
        return None
    return match.group(1).strip()


def _build_tool_response_block(sandbox_result: dict) -> str:
    """Render sandbox output in the training trace format."""
    payload = {
        "result": sandbox_result.get("result", ""),
        "success": bool(sandbox_result.get("success", False)),
    }
    error = str(sandbox_result.get("error", "")).strip()
    if error:
        payload["error"] = error
    return f"<tool_response>{json.dumps(payload, ensure_ascii=True)}</tool_response>"


def _build_rollout_prompt(prompt: object, tool_call_block: str, sandbox_result: dict) -> str:
    """Append the observed tool result and open the follow-up assistant turn."""
    prompt_text = _prompt_to_chatml(prompt).strip()
    tool_response_block = _build_tool_response_block(sandbox_result)
    return (
        f"{prompt_text}\n"
        f"<|im_start|>assistant\n{tool_call_block}<|im_end|>\n"
        f"<|im_start|>tool\n{tool_response_block}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _generate_followup_answer(model, tokenizer, rollout_prompt: str, max_new_tokens: int) -> str:
    """Generate the assistant continuation after the tool result is observed."""
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(rollout_prompt, return_tensors="pt").to(device)
    was_training = model.training
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    try:
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
    finally:
        if was_training:
            model.train()

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return text.split("<|im_end|>", 1)[0].strip()


# ─── reward function for GRPOTrainer ─────────────────────────────────────────

def _build_grpo_reward_fn(
    reward_weights: dict,
    sandbox_timeout: int,
    model,
    tokenizer,
    rollout_max_new_tokens: int,
):
    """
    Build a reward function closure compatible with TRL GRPOTrainer.

    The function signature matches what GRPOTrainer expects:
        fn(prompts, completions, **kwargs) -> list[float]
    """
    from src.training.reward_functions import combined_reward, RewardWeights

    weights = RewardWeights(
        tool_execution=reward_weights.get("tool_execution_weight", 0.5),
        answer_correctness=reward_weights.get("answer_correctness_weight", 0.4),
        format=reward_weights.get("format_weight", 0.1),
    )

    def reward_fn(
        prompts: list[object],
        completions: list[object],
        ground_truths: list[str] | None = None,
        ground_truth: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        gts = ground_truths or ground_truth or kwargs.get("ground_truths") or kwargs.get("ground_truth")
        if not gts:
            gts = [""] * len(completions)

        for index, (completion, gt) in enumerate(zip(completions, gts)):
            completion_text = _completion_to_text(completion)
            prompt = prompts[index] if prompts and index < len(prompts) else ""
            sandbox_result = None

            tool_call_payload = _extract_first_tool_call_payload(completion_text)
            rollout_completion = completion_text

            if tool_call_payload:
                try:
                    parsed = json.loads(tool_call_payload)
                    code = parsed.get("arguments", {}).get("code", "")
                    if code and len(code.strip()) >= 5:
                        sandbox_result = execute_code_sandbox(code, timeout_seconds=sandbox_timeout)
                        tool_call_block = f"<tool_call>{tool_call_payload}</tool_call>"
                        rollout_prompt = _build_rollout_prompt(prompt, tool_call_block, sandbox_result)
                        followup_text = _generate_followup_answer(
                            model,
                            tokenizer,
                            rollout_prompt,
                            max_new_tokens=rollout_max_new_tokens,
                        )
                        rollout_completion = "\n".join(
                            [
                                tool_call_block,
                                _build_tool_response_block(sandbox_result),
                                followup_text,
                            ]
                        ).strip()
                except (json.JSONDecodeError, AttributeError):
                    pass

            r = combined_reward(rollout_completion, gt, sandbox_result, weights)
            rewards.append(r)

        return rewards

    return reward_fn


def _build_grpo_user_prompt(problem: str) -> str:
    """Build a concise tool-first user prompt for GRPO."""
    return (
        "Solve this by using the python_repl tool. On this assistant turn, emit exactly one "
        "<tool_call> JSON block and stop. Do not write verbal reasoning, <tool_response>, or "
        "<final_answer> until the tool result is provided back to you.\n\n"
        f"Problem: {problem}"
    )


def _render_chat_prompt(tokenizer, user_prompt: str) -> str:
    """Render the GRPO user prompt into the model's chat template."""
    system = (
        "You are Genesis Manthan, an AI agent that solves problems by calling tools. "
        "Never reason verbally — always reason through tool execution."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─── training ─────────────────────────────────────────────────────────────────

def _load_grpo_model(model_cfg: dict, lora_cfg: dict, train_cfg: dict, hf_token: str | None):
    """
    Load model for GRPO using Unsloth if available, else standard peft + bitsandbytes.
    Returns (model, tokenizer, used_unsloth: bool).
    """
    import torch
    from transformers import AutoTokenizer

    model_name = model_cfg["name"]
    # Resolve relative local paths to absolute — transformers rejects "./" prefix.
    # HuggingFace Hub IDs have exactly one "/" and never start with "." or contain "\".
    # Anything else is treated as a local filesystem path.
    _is_local = (
        model_name.startswith(".")
        or "\\" in model_name
        or model_name.count("/") != 1
    ) and not Path(model_name).is_absolute()
    if _is_local:
        model_name = str((_PROJECT_ROOT / model_name).resolve())
    max_seq = model_cfg.get("max_seq_length", 1024)
    load_4bit = model_cfg.get("load_in_4bit", True)
    target_modules = lora_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq,
            dtype=torch.float16,
            load_in_4bit=load_4bit,
            token=hf_token,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            target_modules=target_modules,
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=train_cfg.get("seed", 42),
        )
        print("[Manthan] Backend: Unsloth (2× faster)")
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None
        return model, tokenizer, True
    except Exception as e:
        print(f"[Manthan] Unsloth unavailable ({type(e).__name__}: {e})")
        print("[Manthan] Backend: standard transformers + peft + bitsandbytes")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType  # type: ignore[import]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if load_4bit else None,
        dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = False
    # Clear max_length from generation_config to avoid conflict with max_new_tokens
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    # Enable gradient checkpointing for VRAM savings on local GPU
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if getattr(model, "peft_config", None):
        print("[Manthan] Reusing existing PEFT adapter from checkpoint")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        return model, tokenizer, False

    lora_peft = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_peft)
    model.print_trainable_parameters()
    return model, tokenizer, False


def run_grpo_training(cfg: dict, resume_from: str | None = None) -> None:
    """Run Phase 2 GRPO training. Requires GPU. Uses Unsloth if available."""
    import torch
    from datasets import load_dataset, load_from_disk
    from trl import GRPOConfig, GRPOTrainer

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    hub_cfg = cfg.get("hub", {})
    reward_cfg = cfg.get("rewards", {})
    sandbox_cfg = cfg.get("sandbox", {})

    hf_token = os.environ.get("HF_TOKEN")
    is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    device_label = "Kaggle T4" if is_kaggle else "Local GPU"
    print(f"[Manthan] Starting GRPO training on {device_label}")

    if not torch.cuda.is_available():
        print("[Manthan] WARNING: No CUDA device found — GRPO requires a GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[Manthan] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    # Load base model (SFT checkpoint) with fallback
    print(f"[Manthan] Loading model: {model_cfg['name']}")
    model, tokenizer, _ = _load_grpo_model(model_cfg, lora_cfg, train_cfg, hf_token)

    used_vram = torch.cuda.memory_allocated() / 1e9
    print(f"[Manthan] VRAM after model load: {used_vram:.2f} GB")

    # Load reward dataset
    dataset_cfg = cfg.get("dataset", {})
    dataset_path = dataset_cfg.get("path", "shahansha/manthan-tool-reasoning-v1")
    print(f"[Manthan] Loading reward dataset: {dataset_path}")
    dataset_path_obj = Path(dataset_path)
    if dataset_path_obj.exists():
        if dataset_path_obj.is_dir():
            dataset = load_from_disk(str(dataset_path_obj))
            if "train" in dataset:
                dataset = dataset["train"]
        else:
            dataset = load_dataset("json", data_files=str(dataset_path_obj), split="train")
    else:
        try:
            dataset = load_dataset(dataset_path, split="train", token=hf_token)
        except Exception:
            # Fallback to local reward dataset
            local_path = "data/processed/reward_dataset.jsonl"
            print(f"[Manthan] Hub dataset not found, loading from {local_path}")
            dataset = load_dataset("json", data_files=local_path, split="train")

    if "prompt" not in dataset.column_names and "problem" in dataset.column_names:
        dataset = dataset.map(lambda row: {"prompt": _build_grpo_user_prompt(row["problem"])})

    dataset = dataset.map(
        lambda row: {"prompt": _render_chat_prompt(tokenizer, row["prompt"])},
        desc="Rendering GRPO prompts",
    )

    max_samples = dataset_cfg.get("max_samples")
    if max_samples:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))
        print(f"[Manthan] Using reward dataset subset: {len(dataset)} samples")

    # Build reward function
    reward_fn = _build_grpo_reward_fn(
        reward_cfg,
        sandbox_cfg.get("timeout_seconds", 10),
        model,
        tokenizer,
        train_cfg.get("rollout_max_new_tokens", 96),
    )

    # GRPO training config
    output_dir = train_cfg.get("output_dir", "./outputs/grpo_v1")
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        num_generations=train_cfg.get("num_generations", 4),
        max_completion_length=train_cfg.get("max_completion_length", train_cfg.get("max_new_tokens", 512)),
        temperature=train_cfg.get("temperature", 0.9),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        warmup_steps=train_cfg.get("warmup_steps", 20),
        fp16=train_cfg.get("fp16", False),
        seed=train_cfg.get("seed", 42),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 50),
        hub_model_id=hub_cfg.get("hub_model_id") if hub_cfg.get("push_to_hub") else None,
        push_to_hub=hub_cfg.get("push_to_hub", False),
        hub_token=hf_token,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    print(f"[Manthan] Starting GRPO training (min_steps={train_cfg.get('min_steps', 300)})")
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    print(f"[Manthan] Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if hub_cfg.get("push_to_hub") and hf_token:
        hub_id = hub_cfg["hub_model_id"]
        print(f"[Manthan] Pushing final checkpoint to Hub: {hub_id}")
        model.push_to_hub(hub_id, token=hf_token)
        tokenizer.push_to_hub(hub_id, token=hf_token)
        print(f"[Manthan] Uploaded to https://huggingface.co/{hub_id}")


# ─── smoke test ───────────────────────────────────────────────────────────────

def _run_smoke_test() -> None:
    print("Running grpo_train smoke test...")

    # 1. Verify tokenizer loads
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print(f"  OK  tokenizer loaded ({len(tok)} tokens)")

    # 2. Test sandbox
    result = execute_code_sandbox("print(2 + 2)")
    assert result["success"], f"Sandbox failed: {result}"
    assert result["result"] == "4", f"Wrong output: {result['result']}"
    print(f"  OK  sandbox: 2+2={result['result']}")

    # 3. Timeout test
    timeout_result = execute_code_sandbox("import time; time.sleep(20)", timeout_seconds=2)
    assert not timeout_result["success"], "Timeout should fail"
    print(f"  OK  sandbox timeout enforced")

    # 4. Reward functions via smoke import
    from src.training.reward_functions import combined_reward, RewardWeights
    completion = '<tool_call>{"name":"python_repl","arguments":{"code":"print(2+2)"}}</tool_call><final_answer>4</final_answer>'
    r = combined_reward(completion, "4", {"success": True, "result": "4"}, RewardWeights())
    assert 0.0 <= r <= 1.0
    print(f"  OK  combined_reward={r:.2f}")

    print("\nGRPO smoke test PASSED")


# ─── entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan — GRPO Training (Phase 2)")
    parser.add_argument("--smoke-test", action="store_true", help="Validate without training (CPU-safe)")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        if not Path(args.config).exists():
            print(f"Config not found: {args.config}")
            sys.exit(1)
        cfg = _load_config(args.config)
        run_grpo_training(cfg, resume_from=args.resume_from_checkpoint)
