"""
Tool execution success rate metric for Genesis Manthan.
Measures how often tool calls: (1) parse as valid JSON, (2) execute successfully.

Usage:
    python src/eval/tool_success_rate.py --smoke-test
    python src/eval/tool_success_rate.py --model shahansha/Manthan-1.5B
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


PROBE_PROBLEMS = [
    "Calculate 17 * 23 + 45.",
    "What is the square root of 144?",
    "Find all prime numbers less than 30.",
    "Compute the factorial of 10.",
    "What is 15% of 840?",
    "Convert 98.6 degrees Fahrenheit to Celsius.",
    "How many seconds are in a week?",
    "What is the GCD of 48 and 36?",
    "Calculate compound interest: principal=1000, rate=5%, years=3.",
    "Find the median of [5, 2, 8, 1, 9, 3, 7].",
]


def _run_smoke_test() -> None:
    print("Running tool_success_rate smoke test...")

    from src.training.grpo_train import execute_code_sandbox

    # Valid code
    r = execute_code_sandbox("print(17 * 23 + 45)")
    assert r["success"] and r["result"] == "436", f"Failed: {r}"
    print(f"  OK  sandbox: 17*23+45 = {r['result']}")

    # Bad code
    r2 = execute_code_sandbox("import this_module_does_not_exist")
    assert not r2["success"]
    print(f"  OK  sandbox error handling works")

    print("\ntool_success_rate smoke test PASSED")


def evaluate_tool_success(
    model_path: str,
    n_samples: int = 50,
    output_path: str | None = None,
) -> dict:
    """Measure tool call parsability and execution success rate."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.inference.budget_forcing import generate_with_budget_forcing
    from src.training.grpo_train import execute_code_sandbox

    hf_token = os.environ.get("HF_TOKEN")
    print(f"[ToolSuccessRate] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", token=hf_token
    )
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    model.eval()

    # Build problem set (repeat probe problems to reach n_samples)
    problems = (PROBE_PROBLEMS * ((n_samples // len(PROBE_PROBLEMS)) + 1))[:n_samples]

    total = 0
    has_tool_call = 0
    parsable = 0
    executed_success = 0
    results = []

    for i, problem in enumerate(problems):
        generated = generate_with_budget_forcing(model, tokenizer, problem)
        calls = _TOOL_CALL_RE.findall(generated)

        has_call = len(calls) > 0
        is_parsable = False
        did_succeed = False

        if has_call:
            has_tool_call += 1
            try:
                parsed = json.loads(calls[0].strip())
                code = parsed.get("arguments", {}).get("code", "")
                is_parsable = True
                parsable += 1
                if code and len(code.strip()) >= 5:
                    sr = execute_code_sandbox(code)
                    did_succeed = sr["success"]
                    if did_succeed:
                        executed_success += 1
            except (json.JSONDecodeError, AttributeError):
                pass

        total += 1
        results.append({
            "problem": problem,
            "generated": generated,
            "has_tool_call": has_call,
            "parsable": is_parsable,
            "executed_success": did_succeed,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(problems)}] tool_call: {has_tool_call/total:.0%} | "
                  f"parsable: {parsable/total:.0%} | "
                  f"exec_success: {executed_success/total:.0%}")

    summary = {
        "model": model_path,
        "n_samples": total,
        "tool_call_rate": round(has_tool_call / total, 4),
        "parsability_rate": round(parsable / total, 4),
        "execution_success_rate": round(executed_success / total, 4),
        "samples": results,
    }

    print(f"\n[ToolSuccessRate] Results for {model_path}:")
    print(f"  Tool call rate:      {summary['tool_call_rate']:.1%}")
    print(f"  Parsability rate:    {summary['parsability_rate']:.1%}")
    print(f"  Exec success rate:   {summary['execution_success_rate']:.1%}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool success rate metric for Genesis Manthan")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model", type=str, default="shahansha/Manthan-1.5B")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        evaluate_tool_success(
            model_path=args.model,
            n_samples=args.n_samples,
            output_path=args.output,
        )
