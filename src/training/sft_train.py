"""
Phase 1: Supervised Fine-Tuning (SFT) for Genesis Manthan.

Uses Unsloth + TRL SFTTrainer with QLoRA on the Qwen2.5-1.5B base model.
Trains on tool-interaction traces in ChatML format.

Usage:
  Smoke test (CPU, no model load):
    python src/training/sft_train.py --smoke-test

  Local GPU training:
    python src/training/sft_train.py --config configs/sft_config.yaml

  Kaggle:
    python src/training/sft_train.py --config configs/sft_config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Load .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class ModelConfig:
    name: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    max_seq_length: int = 1024
    dtype: str = "float16"
    load_in_4bit: bool = True


@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_steps: int = -1  # -1 means use num_train_epochs; any positive value caps total steps
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    dataloader_num_workers: int = 0
    output_dir: str = "./outputs/sft_v1"


@dataclass
class DatasetConfig:
    train_path: str = "shahansha/manthan-tool-reasoning-v1"
    text_field: str = "text"
    max_samples: Optional[int] = None


@dataclass
class HubConfig:
    push_to_hub: bool = False
    hub_model_id: str = "shahansha/Manthan-1.5B-sft-v0.1"


@dataclass
class SFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    hub: HubConfig = field(default_factory=HubConfig)


def load_config(path: Path) -> SFTConfig:
    """Load SFTConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg_raw = raw.get("sft", raw)  # support both top-level and nested under "sft:"
    cfg = SFTConfig()

    if "model" in cfg_raw:
        for k, v in cfg_raw["model"].items():
            setattr(cfg.model, k, v)
    if "lora" in cfg_raw:
        for k, v in cfg_raw["lora"].items():
            setattr(cfg.lora, k, v)
    if "training" in cfg_raw:
        for k, v in cfg_raw["training"].items():
            setattr(cfg.training, k, v)
    if "dataset" in cfg_raw:
        for k, v in cfg_raw["dataset"].items():
            setattr(cfg.dataset, k, v)
    if "hub" in cfg_raw:
        for k, v in cfg_raw["hub"].items():
            setattr(cfg.hub, k, v)

    return cfg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _load_model_with_fallback(cfg: SFTConfig):
    """
    Load model + tokenizer using Unsloth if available, otherwise fall back to
    standard transformers + peft + bitsandbytes (required on Windows / CPU nodes).

    Returns (model, tokenizer, used_unsloth: bool)
    """
    import torch
    from transformers import AutoTokenizer

    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model.name,
            max_seq_length=cfg.model.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=cfg.model.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            target_modules=cfg.lora.target_modules,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
            random_state=cfg.training.seed,
        )
        print("[Manthan] Backend: Unsloth (2× faster)")
        return model, tokenizer, True
    except Exception as e:
        print(f"[Manthan] Unsloth unavailable ({type(e).__name__}: {e})")
        print("[Manthan] Backend: standard transformers + peft + bitsandbytes")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType  # type: ignore[import]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=bnb_config if cfg.model.load_in_4bit else None,
        dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    # Clear max_new_tokens from generation_config to avoid conflict with SFTConfig max_length
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_new_tokens = None
    # Enable gradient checkpointing for VRAM savings on local GPU
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, False


def train(cfg: SFTConfig) -> None:
    """Run Phase 1 SFT training. Requires GPU. Uses Unsloth if available."""
    import torch
    from trl import SFTTrainer, SFTConfig as TRLSFTConfig  # type: ignore[import]
    from datasets import load_dataset, load_from_disk  # type: ignore[import]

    is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    print(f"Environment: {'Kaggle' if is_kaggle else 'Local GPU'}")
    print(f"Model: {cfg.model.name}")
    print(f"Output: {cfg.training.output_dir}")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. SFT training requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({total_gb:.1f} GB VRAM)")

    # --- Load model (unsloth if available, else standard peft) ---
    model, tokenizer, used_unsloth = _load_model_with_fallback(cfg)

    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after model load: {vram_gb:.1f}GB used / {total_gb:.1f}GB total")

    # --- Load dataset ---
    dataset_path = cfg.dataset.train_path
    if Path(dataset_path).exists():
        print(f"Loading dataset from disk: {dataset_path}")
        from datasets import DatasetDict
        ds = load_from_disk(dataset_path)
        train_dataset = ds["train"] if "train" in ds else ds
        eval_dataset = ds.get("eval", None)
    else:
        print(f"Loading dataset from Hub: {dataset_path}")
        ds = load_dataset(dataset_path)
        train_dataset = ds["train"]
        eval_dataset = ds.get("validation", ds.get("eval", None))

    if cfg.dataset.max_samples:
        train_dataset = train_dataset.select(range(min(cfg.dataset.max_samples, len(train_dataset))))

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")

    # --- Training args (TRL 1.x uses SFTConfig, extends TrainingArguments) ---
    output_dir = cfg.training.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=cfg.training.warmup_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        fp16=cfg.training.fp16,
        bf16=False,  # NEVER bfloat16
        seed=cfg.training.seed,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        report_to="none",
        dataset_text_field=cfg.dataset.text_field,
        max_length=cfg.model.max_seq_length,  # TRL 1.x uses max_length
    )

    # --- SFT Trainer ---
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print("\nStarting SFT training...")
    trainer.train()
    print("Training complete.")

    # --- Save ---
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # --- Push to Hub ---
    if cfg.hub.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("WARNING: HF_TOKEN not set — skipping Hub push")
        else:
            print(f"Pushing to Hub: {cfg.hub.hub_model_id}")
            model.push_to_hub(cfg.hub.hub_model_id, token=hf_token)
            tokenizer.push_to_hub(cfg.hub.hub_model_id, token=hf_token)
            print("Hub push complete.")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _run_smoke_test(cfg: SFTConfig) -> None:
    print("Running sft_train smoke test...\n")

    from transformers import AutoTokenizer  # type: ignore[import]

    # Verify tokenizer loads
    print(f"  Loading tokenizer: Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size} ✓")

    # Print resolved config
    print(f"\n  Config summary:")
    print(f"    Model:          {cfg.model.name}")
    print(f"    LoRA rank:      {cfg.lora.r}")
    print(f"    Seq length:     {cfg.model.max_seq_length}")
    print(f"    Epochs:         {cfg.training.num_train_epochs}")
    print(f"    Batch size:     {cfg.training.per_device_train_batch_size}")
    print(f"    Learning rate:  {cfg.training.learning_rate}")
    print(f"    Output dir:     {cfg.training.output_dir}")
    print(f"    Push to Hub:    {cfg.hub.push_to_hub}")

    # Verify HF_TOKEN is readable (just check env var presence)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"  HF_TOKEN: set ✓")
    else:
        print(f"  HF_TOKEN: NOT SET (set before running full training)")

    print("\nsft_train smoke test PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan SFT training")
    parser.add_argument("--smoke-test", action="store_true", help="Tokenizer-only test, CPU, no training")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_config.yaml"))
    parser.add_argument("--local-rank", type=int, default=0, help="For distributed training")
    args = parser.parse_args()

    cfg = SFTConfig()
    if args.config.exists():
        cfg = load_config(args.config)

    if args.smoke_test:
        _run_smoke_test(cfg)
        sys.exit(0)

    train(cfg)
    sys.exit(0)
