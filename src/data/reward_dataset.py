"""
Reward dataset curator for Genesis Manthan GRPO training.

Pulls from GSM8K, MBPP, and TriviaQA to build a dataset of problems
with verifiable ground-truth answers for GRPO reward checking.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RewardDatasetConfig:
    """Configuration for reward dataset curation."""

    output_path: Path = Path("data/processed/reward_dataset.jsonl")
    n_gsm8k: int = 200
    n_mbpp: int = 150
    n_triviaqa: int = 150
    seed: int = 42


_ANSWER_RE = re.compile(r"####\s*([\-\d,\.]+)")


def build_reward_prompt(problem: str) -> str:
    """Build a concise GRPO user prompt that nudges tool-formatted output."""
    return (
        "Solve this by using the python_repl tool when computation helps. "
        "Keep the response short and return assistant content only in this format:\n"
        '<tool_call>{"name": "python_repl", "arguments": {"code": "print(...)"}}</tool_call>\n'
        '<tool_response>{"result": "...", "success": true}</tool_response>\n'
        '<final_answer>...</final_answer>\n\n'
        f"Problem: {problem}"
    )


def _extract_gsm8k_answer(solution: str) -> str:
    """Extract numeric answer from GSM8K solution string (after #### marker)."""
    match = _ANSWER_RE.search(solution)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def curate_gsm8k(n: int, seed: int) -> list[dict]:
    """Pull n samples from GSM8K test split."""
    from datasets import load_dataset  # type: ignore[import]
    import random

    ds = load_dataset("gsm8k", "main", split="test")
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    records = []
    for i in indices:
        row = ds[i]
        answer = _extract_gsm8k_answer(row["answer"])
        if answer:
            records.append({
                "problem": row["question"],
                "prompt": build_reward_prompt(row["question"]),
                "ground_truth": answer,
                "source": "gsm8k",
                "domain": "math",
            })
    return records


def curate_mbpp(n: int, seed: int) -> list[dict]:
    """Pull n samples from MBPP test split."""
    from datasets import load_dataset  # type: ignore[import]
    import random

    ds = load_dataset("mbpp", "sanitized", split="test")
    random.seed(seed + 1)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    records = []
    for i in indices:
        row = ds[i]
        problem = row.get("text") or row.get("prompt")
        if not problem:
            available_columns = ", ".join(sorted(row.keys()))
            raise KeyError(f"MBPP row missing problem text field; available columns: {available_columns}")

        # For MBPP, ground truth is whether the code passes test_list
        # We store the canonical solution as reference
        records.append({
            "problem": problem,
            "prompt": build_reward_prompt(problem),
            "ground_truth": row["code"],
            "test_list": row.get("test_list", []),
            "source": "mbpp",
            "domain": "code",
        })
    return records


def curate_triviaqa(n: int, seed: int) -> list[dict]:
    """Pull n samples from TriviaQA validation split."""
    from datasets import load_dataset  # type: ignore[import]
    import random

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    random.seed(seed + 2)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    records = []
    for i in indices:
        row = ds[i]
        aliases = row["answer"].get("aliases", [])
        normalized = row["answer"].get("normalized_value", "")
        if normalized:
            records.append({
                "problem": row["question"],
                "prompt": build_reward_prompt(row["question"]),
                "ground_truth": normalized,
                "answer_aliases": aliases[:5],  # keep up to 5 aliases for fuzzy matching
                "source": "triviaqa",
                "domain": "factual",
            })
    return records


def build_reward_dataset(config: RewardDatasetConfig) -> None:
    """Build and save the reward dataset as JSONL."""
    print("Curating GSM8K samples...")
    gsm8k = curate_gsm8k(config.n_gsm8k, config.seed)
    print(f"  GSM8K: {len(gsm8k)} samples")

    print("Curating MBPP samples...")
    mbpp = curate_mbpp(config.n_mbpp, config.seed)
    print(f"  MBPP: {len(mbpp)} samples")

    print("Curating TriviaQA samples...")
    triviaqa = curate_triviaqa(config.n_triviaqa, config.seed)
    print(f"  TriviaQA: {len(triviaqa)} samples")

    all_records = gsm8k + mbpp + triviaqa
    print(f"\nTotal: {len(all_records)} reward samples")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8") as fout:
        for rec in all_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved to {config.output_path}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _run_smoke_test() -> None:
    print("Running reward_dataset smoke test...\n")

    # Test GSM8K answer extraction
    test_solution = "She earns $120. #### 120"
    answer = _extract_gsm8k_answer(test_solution)
    assert answer == "120", f"GSM8K extraction failed: got '{answer}'"
    print("  GSM8K answer extraction: PASS ✓")

    test_solution_negative = "He lost $45. #### -45"
    answer2 = _extract_gsm8k_answer(test_solution_negative)
    assert answer2 == "-45", f"GSM8K negative extraction failed: got '{answer2}'"
    print("  GSM8K negative answer extraction: PASS ✓")

    # Test config
    cfg = RewardDatasetConfig()
    assert cfg.n_gsm8k + cfg.n_mbpp + cfg.n_triviaqa == 500
    print("  RewardDatasetConfig default (500 total): PASS ✓")

    print("\nreward_dataset smoke test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan reward dataset curator")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/reward_dataset.jsonl"))
    parser.add_argument("--n-gsm8k", type=int, default=200)
    parser.add_argument("--n-mbpp", type=int, default=150)
    parser.add_argument("--n-triviaqa", type=int, default=150)
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        sys.exit(0)

    cfg = RewardDatasetConfig(
        output_path=args.output,
        n_gsm8k=args.n_gsm8k,
        n_mbpp=args.n_mbpp,
        n_triviaqa=args.n_triviaqa,
    )
    build_reward_dataset(cfg)
