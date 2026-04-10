"""
MBPP pass@1 evaluation for Genesis Manthan.
Tests tool-mediated code generation on the MBPP benchmark.

Usage:
    python src/eval/benchmark_mbpp.py --smoke-test
    python src/eval/benchmark_mbpp.py --model shahansha/Manthan-1.5B --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _get_problem_text(sample: dict) -> str:
    """Return the MBPP prompt text across dataset schema variants."""
    problem = sample.get("text") or sample.get("prompt")
    if not problem:
        available_columns = ", ".join(sorted(sample.keys()))
        raise KeyError(f"MBPP sample missing problem text field; available columns: {available_columns}")
    return problem


def _extract_code_from_completion(text: str) -> str | None:
    """Extract Python code from tool_call, final_answer, or code blocks."""
    # Try tool_call first
    calls = _TOOL_CALL_RE.findall(text)
    if calls:
        try:
            parsed = json.loads(calls[-1].strip())  # last tool call = likely the solution
            code = parsed.get("arguments", {}).get("code", "")
            if code:
                return code
        except (json.JSONDecodeError, AttributeError):
            pass

    # Try final_answer code block
    ans = _FINAL_ANSWER_RE.search(text)
    if ans:
        cb = _CODE_BLOCK_RE.search(ans.group(1))
        if cb:
            return cb.group(1)

    # Bare code block anywhere
    cb = _CODE_BLOCK_RE.search(text)
    if cb:
        return cb.group(1)

    return None


def _run_test_case(code: str, test: str, timeout: int = 10) -> bool:
    """Execute code + test case in a sandboxed subprocess. Returns True if test passes."""
    full_code = f"{code}\n\n{test}"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
        dir=tempfile.gettempdir(), prefix="manthan_mbpp_",
    ) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _run_smoke_test() -> None:
    print("Running benchmark_mbpp smoke test...")
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    sample = ds[0]
    problem = _get_problem_text(sample)
    print(f"  OK  MBPP loaded — sample task_id: {sample['task_id']}")
    print(f"  OK  Task: {problem[:60]}...")

    # Test sandbox execution
    code = "def add(a, b):\n    return a + b"
    test = "assert add(2, 3) == 5"
    passed = _run_test_case(code, test)
    assert passed, "Basic test case should pass"
    print("  OK  Sandbox test execution works")

    fail_code = "def add(a, b):\n    return a - b"
    assert not _run_test_case(fail_code, test), "Wrong code should fail"
    print("  OK  Sandbox correctly fails wrong code")

    print("\nMBPP smoke test PASSED")


def evaluate_mbpp(
    model_path: str,
    n_samples: int = 100,
    use_budget_forcing: bool = True,
    output_path: str | None = None,
) -> dict:
    """Evaluate Manthan on MBPP pass@1."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from src.inference.budget_forcing import generate_with_budget_forcing

    hf_token = os.environ.get("HF_TOKEN")
    print(f"[MBPP] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", token=hf_token
    )
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    model.eval()

    print("[MBPP] Loading dataset...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    if n_samples > 0:
        ds = ds.select(range(min(n_samples, len(ds))))

    passed_total = 0
    results = []

    for i, sample in enumerate(ds):
        problem = f"Write a Python function: {_get_problem_text(sample)}"
        test_cases = sample.get("test_list", [])

        t0 = time.time()
        if use_budget_forcing:
            generated = generate_with_budget_forcing(model, tokenizer, problem)
        else:
            inputs = tokenizer(problem, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512)
            generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        elapsed = time.time() - t0

        code = _extract_code_from_completion(generated)
        pass_count = 0
        if code and test_cases:
            for test in test_cases:
                if _run_test_case(code, test):
                    pass_count += 1

        passed = pass_count == len(test_cases) and len(test_cases) > 0
        if passed:
            passed_total += 1

        results.append({
            "task_id": sample["task_id"],
            "problem": problem,
            "generated": generated,
            "extracted_code": code,
            "passed": passed,
            "pass_count": pass_count,
            "total_tests": len(test_cases),
            "elapsed_s": round(elapsed, 2),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ds)}] pass@1: {passed_total/(i+1):.1%}")

    summary = {
        "model": model_path,
        "n_samples": len(ds),
        "pass_at_1": round(passed_total / len(ds), 4),
        "samples": results,
    }

    print(f"\n[MBPP] pass@1: {summary['pass_at_1']:.1%}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBPP evaluation for Genesis Manthan")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model", type=str, default="shahansha/Manthan-1.5B")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--no-budget-forcing", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        evaluate_mbpp(
            model_path=args.model,
            n_samples=args.n_samples,
            use_budget_forcing=not args.no_budget_forcing,
            output_path=args.output,
        )
