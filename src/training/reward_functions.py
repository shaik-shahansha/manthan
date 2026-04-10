"""
Composable reward functions for Genesis Manthan GRPO training.

Each function returns a float in [0.0, 1.0] and handles None/malformed inputs gracefully.
Never combine rewards inside a single function — keep them composable.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """Weights for combining individual reward signals."""

    tool_execution: float = 0.5
    answer_correctness: float = 0.4
    format: float = 0.1

    def __post_init__(self) -> None:
        total = self.tool_execution + self.answer_correctness + self.format
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Reward weights must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|<tool_response>|<final_answer>|$)", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)


def _extract_tool_calls(text: str) -> list[str]:
    """Return raw content strings from all <tool_call> blocks."""
    return _TOOL_CALL_RE.findall(text)


def _extract_final_answer(text: str) -> str | None:
    """Return the content of the first <final_answer> block, or None."""
    match = _FINAL_ANSWER_RE.search(text)
    return match.group(1).strip() if match else None


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _parse_number(text: str) -> float | None:
    """Extract the first numeric value from a string, or None."""
    text = text.strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Reward function 1: Tool execution
# ---------------------------------------------------------------------------


def tool_execution_reward(completion: str, sandbox_result: dict | None = None) -> float:
    """
    Reward based on tool call quality and execution success.

    Scoring:
      - 0.0  : no <tool_call> block found
      - 0.0  : tool_call content is invalid JSON
      - 0.0  : tool_call JSON parses but code field is empty / < 5 chars
      - 0.5  : tool_call is valid JSON with non-trivial code
      - 0.25 : execution ran (sandbox_result provided) but produced empty output
      - 1.0  : execution succeeded with non-empty output

    Args:
        completion: Raw model completion string.
        sandbox_result: Dict with keys "success" (bool) and "result" (str), or None.

    Returns:
        Float reward in [0.0, 1.0].
    """
    if not completion:
        return 0.0

    calls = _extract_tool_calls(completion)
    if not calls:
        return 0.0

    # Validate the first tool call JSON
    raw = calls[0].strip()
    if not _is_valid_json(raw):
        return 0.0

    parsed = json.loads(raw)
    code = ""
    if isinstance(parsed, dict):
        args = parsed.get("arguments", {})
        if isinstance(args, dict):
            code = args.get("code", "")

    if len(code.strip()) < 5:
        return 0.0

    # JSON is valid and code is non-trivial
    if sandbox_result is None:
        return 0.5

    if not sandbox_result.get("success", False):
        return 0.25

    result_str = str(sandbox_result.get("result", "")).strip()
    if not result_str:
        return 0.25

    return 1.0


# ---------------------------------------------------------------------------
# Reward function 2: Answer correctness
# ---------------------------------------------------------------------------


def answer_correctness_reward(completion: str, ground_truth: str) -> float:
    """
    Reward based on correctness of the <final_answer> against ground truth.

    Scoring:
      - 0.0  : no <final_answer> tag found
      - 0.0  : ground_truth is empty
      - 1.0  : exact string match (case-insensitive, stripped)
      - 1.0  : numeric match within 0.1% tolerance
      - 0.5  : numeric match within 1% tolerance
      - 0.0  : otherwise

    Args:
        completion: Raw model completion string.
        ground_truth: Expected answer string.

    Returns:
        Float reward in [0.0, 1.0].
    """
    if not ground_truth or ground_truth.strip() == "":
        return 0.0

    predicted = _extract_final_answer(completion)
    if predicted is None:
        return 0.0

    # Exact match
    if predicted.lower() == ground_truth.strip().lower():
        return 1.0

    # Numeric comparison
    pred_num = _parse_number(predicted)
    truth_num = _parse_number(ground_truth)

    if pred_num is not None and truth_num is not None and truth_num != 0:
        relative_error = abs(pred_num - truth_num) / abs(truth_num)
        if relative_error <= 0.001:
            return 1.0
        if relative_error <= 0.01:
            return 0.5

    return 0.0


# ---------------------------------------------------------------------------
# Reward function 3: Format reward
# ---------------------------------------------------------------------------


def format_reward(completion: str) -> float:
    """
    Reward for using the tool-mediated format.

    Returns 1.0 if the completion contains at least one <tool_call> block,
    0.0 otherwise. Weighting is applied by RewardWeights at combination time.

    Args:
        completion: Raw model completion string.

    Returns:
        1.0 or 0.0.
    """
    if not completion:
        return 0.0
    return 1.0 if "<tool_call>" in completion else 0.0


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------


def combined_reward(
    completion: str,
    ground_truth: str,
    sandbox_result: dict | None = None,
    weights: RewardWeights | None = None,
) -> float:
    """
    Weighted combination of all three reward signals, clipped to [0.0, 1.0].

    Args:
        completion: Raw model completion string.
        ground_truth: Expected answer string.
        sandbox_result: Sandbox execution result dict, or None.
        weights: RewardWeights instance (uses defaults if None).

    Returns:
        Float in [0.0, 1.0].
    """
    if weights is None:
        weights = RewardWeights()

    r_tool = tool_execution_reward(completion, sandbox_result)
    r_correct = answer_correctness_reward(completion, ground_truth)
    r_format = format_reward(completion)

    score = (
        weights.tool_execution * r_tool
        + weights.answer_correctness * r_correct
        + weights.format * r_format
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_MOCK_CASES = [
    {
        "name": "valid tool call, correct answer",
        "completion": (
            '<tool_call>{"name":"python_repl","arguments":{"code":"print(17*23)"}}</tool_call>\n'
            "<tool_response>{\"result\":\"391\",\"success\":true}</tool_response>\n"
            "<final_answer>391</final_answer>"
        ),
        "ground_truth": "391",
        "sandbox_result": {"success": True, "result": "391"},
    },
    {
        "name": "no tool call (verbal CoT)",
        "completion": "The answer is 391 because 17 times 23 equals 391.",
        "ground_truth": "391",
        "sandbox_result": None,
    },
    {
        "name": "tool call with invalid JSON",
        "completion": "<tool_call>not_json_at_all</tool_call><final_answer>391</final_answer>",
        "ground_truth": "391",
        "sandbox_result": None,
    },
    {
        "name": "tool call valid JSON, wrong answer",
        "completion": (
            '<tool_call>{"name":"python_repl","arguments":{"code":"print(17*22)"}}</tool_call>\n'
            "<tool_response>{\"result\":\"374\",\"success\":true}</tool_response>\n"
            "<final_answer>374</final_answer>"
        ),
        "ground_truth": "391",
        "sandbox_result": {"success": True, "result": "374"},
    },
    {
        "name": "no final_answer tag",
        "completion": (
            '<tool_call>{"name":"python_repl","arguments":{"code":"print(17*23)"}}</tool_call>\n'
            "<tool_response>{\"result\":\"391\",\"success\":true}</tool_response>"
        ),
        "ground_truth": "391",
        "sandbox_result": {"success": True, "result": "391"},
    },
]


def _run_smoke_test() -> None:
    print("Running reward_functions smoke test...\n")
    all_passed = True

    for case in _MOCK_CASES:
        r_tool = tool_execution_reward(case["completion"], case["sandbox_result"])
        r_correct = answer_correctness_reward(case["completion"], case["ground_truth"])
        r_fmt = format_reward(case["completion"])
        r_combined = combined_reward(
            case["completion"], case["ground_truth"], case["sandbox_result"]
        )

        # All outputs must be in [0.0, 1.0]
        for name, val in [
            ("tool_execution", r_tool),
            ("answer_correctness", r_correct),
            ("format", r_fmt),
            ("combined", r_combined),
        ]:
            if not (0.0 <= val <= 1.0):
                print(f"  FAIL [{case['name']}] {name}={val} out of range!")
                all_passed = False

        print(
            f"  [{case['name']}]\n"
            f"    tool={r_tool:.2f}  correct={r_correct:.2f}  "
            f"format={r_fmt:.2f}  combined={r_combined:.2f}"
        )

    print()
    if all_passed:
        print("reward_functions smoke test PASSED")
    else:
        print("reward_functions smoke test FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan reward functions")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests and exit")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        print("reward_functions.py — import this module; use --smoke-test to run tests")
