"""
Dataset formatter for Genesis Manthan.

Converts raw JSONL tool-interaction traces into a HuggingFace Dataset
in ChatML format, ready for Unsloth SFT training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are Genesis Manthan, an AI agent that solves problems by calling tools. "
    "Never reason verbally — always reason through tool execution."
)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)


@dataclass
class FormatConfig:
    """Configuration for dataset formatting."""

    input_path: Path = Path("data/raw/synthetic_traces.jsonl")
    output_path: Path = Path("data/processed/manthan_dataset")
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 1024
    eval_fraction: float = 0.1
    seed: int = 42


# ---------------------------------------------------------------------------
# ChatML conversion
# ---------------------------------------------------------------------------


def trace_to_chatml(problem: str, trace: str) -> str:
    """
    Convert a problem + trace string to ChatML format.

    The trace must contain <tool_call>, <tool_response>, and <final_answer> blocks.
    Tool calls and responses are interleaved as separate assistant/tool turns.
    The final_answer is the last assistant turn.

    Returns:
        A single ChatML string with <|im_start|>/<|im_end|> tokens.
    """
    calls = _TOOL_CALL_RE.findall(trace)
    responses = _TOOL_RESPONSE_RE.findall(trace)
    final_answers = _FINAL_ANSWER_RE.findall(trace)

    final_answer = final_answers[0].strip() if final_answers else ""

    parts: list[str] = []

    # System turn
    parts.append(f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>")

    # User turn
    parts.append(f"<|im_start|>user\n{problem.strip()}<|im_end|>")

    # Interleaved tool calls and responses
    for i, call_content in enumerate(calls):
        call_block = f"<tool_call>{call_content.strip()}</tool_call>"
        parts.append(f"<|im_start|>assistant\n{call_block}<|im_end|>")

        if i < len(responses):
            resp_block = f"<tool_response>{responses[i].strip()}</tool_response>"
            parts.append(f"<|im_start|>tool\n{resp_block}<|im_end|>")

    # Final assistant answer
    if final_answer:
        parts.append(f"<|im_start|>assistant\n<final_answer>{final_answer}</final_answer><|im_end|>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_record(record: dict) -> tuple[bool, str]:
    """
    Validate a raw JSONL record has the required fields and trace structure.

    Returns:
        Tuple of (is_valid, reason).
    """
    if "problem" not in record:
        return False, "Missing 'problem' field"
    if "trace" not in record:
        return False, "Missing 'trace' field"

    trace = record["trace"]

    calls = _TOOL_CALL_RE.findall(trace)
    responses = _TOOL_RESPONSE_RE.findall(trace)
    answers = _FINAL_ANSWER_RE.findall(trace)

    if not calls:
        return False, "No <tool_call> blocks found"
    if not responses:
        return False, "No <tool_response> blocks found"
    if not answers:
        return False, "No <final_answer> block found"

    for i, call_content in enumerate(calls):
        try:
            json.loads(call_content.strip())
        except json.JSONDecodeError as exc:
            return False, f"tool_call[{i}] is not valid JSON: {exc}"

    for i, resp_content in enumerate(responses):
        try:
            parsed = json.loads(resp_content.strip())
            if "success" not in parsed:
                return False, f"tool_response[{i}] missing 'success' field"
        except json.JSONDecodeError as exc:
            return False, f"tool_response[{i}] is not valid JSON: {exc}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_dataset(config: FormatConfig) -> None:
    """
    Load raw JSONL, validate, convert to ChatML, filter by token count,
    split train/eval, and save as a HuggingFace DatasetDict.
    """
    from datasets import Dataset, DatasetDict  # type: ignore[import]
    from transformers import AutoTokenizer  # type: ignore[import]

    print(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Load raw records
    raw_records: list[dict] = []
    with open(config.input_path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(raw_records)} raw records from {config.input_path}")

    # Validate and convert
    texts: list[str] = []
    token_counts: list[int] = []
    filtered_invalid = 0
    filtered_long = 0

    for rec in raw_records:
        is_valid, reason = validate_record(rec)
        if not is_valid:
            filtered_invalid += 1
            continue

        chatml = trace_to_chatml(rec["problem"], rec["trace"])
        tokens = tokenizer(chatml, return_tensors=None)["input_ids"]
        n_tokens = len(tokens)

        if n_tokens > config.max_tokens:
            filtered_long += 1
            continue

        texts.append(chatml)
        token_counts.append(n_tokens)

    print(f"\nFiltered: {filtered_invalid} invalid, {filtered_long} too long (>{config.max_tokens} tokens)")
    print(f"Kept: {len(texts)} samples")

    if not texts:
        print("ERROR: No samples passed validation. Check input data.")
        sys.exit(1)

    # Token count statistics
    sorted_counts = sorted(token_counts)
    n = len(sorted_counts)
    p50 = sorted_counts[int(n * 0.50)]
    p90 = sorted_counts[int(n * 0.90)]
    p95 = sorted_counts[int(n * 0.95)]
    print(f"\nToken distribution: p50={p50}  p90={p90}  p95={p95}  max={max(sorted_counts)}")

    # Train/eval split
    import random
    random.seed(config.seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)

    n_eval = max(1, int(len(texts) * config.eval_fraction))
    eval_indices = set(indices[:n_eval])

    train_texts = [texts[i] for i in range(len(texts)) if i not in eval_indices]
    eval_texts = [texts[i] for i in eval_indices]

    print(f"\nSplit: {len(train_texts)} train / {len(eval_texts)} eval")

    # Build and save HuggingFace DatasetDict
    config.output_path.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_texts}),
            "eval": Dataset.from_dict({"text": eval_texts}),
        }
    )
    dataset_dict.save_to_disk(str(config.output_path))
    print(f"\nDataset saved to {config.output_path}")
    print("Fields: 'text' (ChatML string)")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_SAMPLE_RECORD = {
    "problem": "What is 17 × 23?",
    "trace": (
        '<tool_call>{"name": "python_repl", "arguments": {"code": "print(17 * 23)"}}</tool_call>\n'
        '<tool_response>{"result": "391", "success": true}</tool_response>\n'
        "<final_answer>391</final_answer>"
    ),
    "source": "synthetic",
    "domain": "math",
}


def _run_smoke_test() -> None:
    print("Running format_dataset smoke test...\n")

    # Validate a good record
    ok, reason = validate_record(_SAMPLE_RECORD)
    assert ok, f"Sample record failed validation: {reason}"
    print("  validate_record (valid): PASS ✓")

    # Validate a bad record (missing final_answer)
    bad_record = {
        "problem": "X?",
        "trace": '<tool_call>{"name":"python_repl","arguments":{"code":"print(1)"}}</tool_call>\n',
    }
    ok2, reason2 = validate_record(bad_record)
    assert not ok2, "Bad record should fail validation"
    print(f"  validate_record (invalid, missing final_answer): PASS ✓")

    # Convert to ChatML
    chatml = trace_to_chatml(_SAMPLE_RECORD["problem"], _SAMPLE_RECORD["trace"])
    assert "<|im_start|>system" in chatml
    assert "<|im_start|>user" in chatml
    assert "<tool_call>" in chatml
    assert "<tool_response>" in chatml
    assert "<final_answer>" in chatml
    assert "<|im_end|>" in chatml
    print("  trace_to_chatml: PASS ✓")

    # Tokenizer load (CPU, no GPU)
    from transformers import AutoTokenizer  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokens = tokenizer(chatml)["input_ids"]
    print(f"  Tokenizer load + encode sample: {len(tokens)} tokens ✓")

    assert len(tokens) < 1024, f"Sample trace too long: {len(tokens)} tokens"
    print(f"  Token count < 1024: PASS ✓")

    print("\nformat_dataset smoke test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan dataset formatter")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests and exit")
    parser.add_argument("--input", type=Path, default=Path("data/raw/synthetic_traces.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/manthan_dataset"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        sys.exit(0)

    cfg = FormatConfig(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
    )
    build_dataset(cfg)
