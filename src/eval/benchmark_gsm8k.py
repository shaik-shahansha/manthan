"""
GSM8K benchmark evaluation for Genesis Manthan.
Evaluates tool-augmented reasoning accuracy on grade-school math problems.

Usage:
    python src/eval/benchmark_gsm8k.py --smoke-test
    python src/eval/benchmark_gsm8k.py --model shahansha/Manthan-1.5B --n-samples 100
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


_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_GSM8K_ANSWER_RE = re.compile(r"#### (-?\d+(?:,\d{3})*(?:\.\d+)?)")
# Fallback: extract last standalone number from plain text (handles verbal answers)
_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?")


def _extract_final_answer(text: str) -> str | None:
    """Extract answer from <final_answer> tag. Returns None if tag absent."""
    m = _FINAL_ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_answer_fallback(text: str) -> str | None:
    """
    Fallback: find the last standalone number in the completion.
    Used when the model answers verbally without a <final_answer> tag.
    """
    # Also check for numbers in <tool_response> result fields
    tool_resp_re = re.compile(r'<tool_response>\s*(.*?)\s*</tool_response>', re.DOTALL)
    for m in tool_resp_re.finditer(text):
        try:
            import json as _j
            resp = _j.loads(m.group(1))
            result = str(resp.get("result", "")).strip()
            nums = _NUMBER_RE.findall(result)
            if nums:
                return nums[-1].replace(",", "")
        except Exception:
            pass

    # Find last number in the whole completion
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _extract_gsm8k_answer(solution: str) -> str | None:
    """Extract the numeric answer from GSM8K solution field (after ####)."""
    m = _GSM8K_ANSWER_RE.search(solution)
    if m:
        return m.group(1).replace(",", "")
    return None


def _answers_match(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    # Clean both
    pred_clean = re.sub(r"[,$%\s]", "", predicted.lower())
    gt_clean = re.sub(r"[,$%\s]", "", ground_truth.lower())
    if pred_clean == gt_clean:
        return True
    # Numeric comparison
    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        return abs(pred_num - gt_num) / max(abs(gt_num), 1e-9) < 0.001
    except ValueError:
        return False


def _run_smoke_test() -> None:
    print("Running benchmark_gsm8k smoke test...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    sample = ds[0]
    gt = _extract_gsm8k_answer(sample["answer"])
    print(f"  OK  GSM8K loaded — sample problem: {sample['question'][:60]}...")
    print(f"  OK  Ground truth parsed: {gt}")

    # Test answer matching
    assert _answers_match("72", "72"), "Exact match failed"
    assert _answers_match("$72.00", "72"), "Dollar sign match failed"
    assert not _answers_match("73", "72"), "Wrong answer should not match"
    print("  OK  Answer matching logic correct")
    print("\nGSM8K smoke test PASSED")


def evaluate_gsm8k(
    model_path: str,
    n_samples: int = 100,
    use_budget_forcing: bool = True,
    min_tool_calls: int = 1,
    max_tool_calls: int = 5,
    output_path: str | None = None,
) -> dict:
    """
    Evaluate Manthan on GSM8K test set.

    Returns:
        dict with accuracy, tool_call_parsability, avg_tool_calls, timeout_rate, samples
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from src.inference.budget_forcing import generate_with_budget_forcing

    hf_token = os.environ.get("HF_TOKEN")
    print(f"[GSM8K] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    model.eval()

    print("[GSM8K] Loading dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples > 0:
        ds = ds.select(range(min(n_samples, len(ds))))

    print(f"[GSM8K] Evaluating {len(ds)} samples...")

    results = []
    correct = 0
    parsable = 0
    tool_call_counts = []
    timeouts = 0

    import json as _json
    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    for i, sample in enumerate(ds):
        problem = sample["question"]
        gt = _extract_gsm8k_answer(sample["answer"]) or ""

        t0 = time.time()
        if use_budget_forcing:
            generated = generate_with_budget_forcing(
                model, tokenizer, problem, min_calls=min_tool_calls, max_calls=max_tool_calls
            )
        else:
            from src.inference.budget_forcing import _build_prompt
            prompt_text = _build_prompt(tokenizer, problem)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        elapsed = time.time() - t0

        predicted = _extract_final_answer(generated)
        # Strict: only count <final_answer> tag answers
        is_correct_strict = _answers_match(predicted, gt)

        # Fallback: also try extracting number from tool_response or plain text
        predicted_fallback = predicted or _extract_answer_fallback(generated)
        is_correct_fallback = _answers_match(predicted_fallback, gt)

        # Count correct using fallback (more fair metric during early training)
        is_correct = is_correct_fallback

        # Count tool calls
        calls = _TOOL_CALL_RE.findall(generated)
        tool_call_count = len(calls)
        tool_call_counts.append(tool_call_count)

        # Check parsability of first tool call
        is_parsable = False
        if calls:
            try:
                _json.loads(calls[0].strip())
                is_parsable = True
            except (ValueError, _json.JSONDecodeError):
                pass

        if is_correct:
            correct += 1
        if is_correct_strict:
            pass  # tracked per-sample below
        if is_parsable or tool_call_count == 0:
            parsable += 1
        if elapsed > 15:
            timeouts += 1

        results.append({
            "problem": problem,
            "ground_truth": gt,
            "predicted": predicted,
            "predicted_fallback": predicted_fallback,
            "generated": generated,
            "correct": is_correct,
            "correct_strict": is_correct_strict,
            "tool_calls": tool_call_count,
            "parsable": is_parsable,
            "elapsed_s": round(elapsed, 2),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ds)}] Accuracy: {correct/(i+1):.1%} | "
                  f"Parsability: {parsable/(i+1):.1%} | "
                  f"Avg tool calls: {sum(tool_call_counts)/len(tool_call_counts):.1f}")

    correct_strict = sum(1 for r in results if r["correct_strict"])
    summary = {
        "model": model_path,
        "n_samples": len(ds),
        "accuracy": round(correct / len(ds), 4),              # fallback (fair) accuracy
        "accuracy_strict": round(correct_strict / len(ds), 4), # only <final_answer> tag
        "tool_call_parsability": round(parsable / len(ds), 4),
        "avg_tool_calls_per_problem": round(sum(tool_call_counts) / len(tool_call_counts), 2),
        "timeout_rate": round(timeouts / len(ds), 4),
        "samples": results,
    }

    print(f"\n[GSM8K] Final Results:")
    print(f"  Accuracy (fallback): {summary['accuracy']:.1%}  (includes verbal numeric answers)")
    print(f"  Accuracy (strict):   {summary['accuracy_strict']:.1%}  (requires <final_answer> tag)")
    print(f"  Tool parsability:    {summary['tool_call_parsability']:.1%}")
    print(f"  Avg tool calls:      {summary['avg_tool_calls_per_problem']:.2f}")
    print(f"  Timeout rate:        {summary['timeout_rate']:.1%}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K evaluation for Genesis Manthan")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model", type=str, default="shahansha/Manthan-1.5B")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--no-budget-forcing", action="store_true")
    parser.add_argument("--min-tool-calls", type=int, default=1)
    parser.add_argument("--max-tool-calls", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        evaluate_gsm8k(
            model_path=args.model,
            n_samples=args.n_samples,
            use_budget_forcing=not args.no_budget_forcing,
            min_tool_calls=args.min_tool_calls,
            max_tool_calls=args.max_tool_calls,
            output_path=args.output,
        )
