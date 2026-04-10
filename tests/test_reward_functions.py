"""Unit tests for reward functions — all run on CPU, no GPU required."""

from __future__ import annotations

import pytest
from src.training.reward_functions import (
    RewardWeights,
    answer_correctness_reward,
    combined_reward,
    format_reward,
    tool_execution_reward,
)

GOOD = (
    '<tool_call>{"name": "python_repl", "arguments": {"code": "print(17*23)"}}</tool_call>\n'
    '<tool_response>{"result": "391", "success": true}</tool_response>\n'
    "<final_answer>391</final_answer>"
)
NO_TOOL = "The answer is 391 because 17 times 23 equals 391."
BAD_JSON = "<tool_call>{not valid json!}</tool_call><final_answer>391</final_answer>"
EMPTY_CODE = '<tool_call>{"name": "python_repl", "arguments": {"code": ""}}</tool_call><final_answer>4</final_answer>'
PARTIAL_TOOL_CALL = (
    '<tool_call>{"name": "python_repl", "arguments": {"code": "print(17*23)"}}\n\n'
    '<tool_response>{"result": "391", "success": true}</tool_response>\n'
    '<final_answer>391</final_answer>'
)

SUCCESS_SANDBOX = {"success": True, "result": "391"}
FAIL_SANDBOX = {"success": False, "result": "", "error": "SyntaxError"}
EMPTY_RESULT_SANDBOX = {"success": True, "result": ""}


# ─── tool_execution_reward ────────────────────────────────────────────────────

class TestToolExecutionReward:
    def test_no_tool_call_returns_zero(self):
        assert tool_execution_reward(NO_TOOL, None) == 0.0

    def test_empty_string_returns_zero(self):
        assert tool_execution_reward("", None) == 0.0

    def test_none_input_handled(self):
        assert tool_execution_reward(None, None) == 0.0

    def test_bad_json_returns_zero(self):
        assert tool_execution_reward(BAD_JSON, None) == 0.0

    def test_empty_code_returns_zero(self):
        # empty code field is treated as non-trivial → returns 0.0
        assert tool_execution_reward(EMPTY_CODE, None) == 0.0

    def test_valid_call_no_sandbox_returns_half(self):
        assert tool_execution_reward(GOOD, None) == 0.5

    def test_valid_call_success_sandbox_returns_one(self):
        assert tool_execution_reward(GOOD, SUCCESS_SANDBOX) == 1.0

    def test_partial_tool_call_before_tool_response_is_handled(self):
        assert tool_execution_reward(PARTIAL_TOOL_CALL, SUCCESS_SANDBOX) == 1.0

    def test_valid_call_fail_sandbox_returns_quarter(self):
        assert tool_execution_reward(GOOD, FAIL_SANDBOX) == 0.25

    def test_empty_result_sandbox_returns_partial(self):
        # success=True but empty result string → 0.25 (ran but no output)
        assert tool_execution_reward(GOOD, EMPTY_RESULT_SANDBOX) == 0.25

    def test_result_always_in_range(self):
        for completion in [GOOD, NO_TOOL, BAD_JSON, EMPTY_CODE, "", None]:
            for sandbox in [None, SUCCESS_SANDBOX, FAIL_SANDBOX]:
                r = tool_execution_reward(completion, sandbox)
                assert 0.0 <= r <= 1.0


# ─── answer_correctness_reward ────────────────────────────────────────────────

class TestAnswerCorrectnessReward:
    def test_exact_match(self):
        assert answer_correctness_reward(GOOD, "391") == 1.0

    def test_case_insensitive_exact(self):
        c = "<final_answer>Paris</final_answer>"
        assert answer_correctness_reward(c, "paris") == 1.0

    def test_numeric_near_match(self):
        c = "<final_answer>391.001</final_answer>"
        assert answer_correctness_reward(c, "391") >= 0.5

    def test_wrong_answer(self):
        assert answer_correctness_reward(GOOD, "400") == 0.0

    def test_no_final_answer_tag(self):
        assert answer_correctness_reward(NO_TOOL, "391") == 0.0

    def test_empty_ground_truth(self):
        assert answer_correctness_reward(GOOD, "") == 0.0

    def test_dollar_amount(self):
        c = "<final_answer>$391</final_answer>"
        assert answer_correctness_reward(c, "391") >= 0.9

    def test_result_always_in_range(self):
        for gt in ["391", "0", "Paris", ""]:
            r = answer_correctness_reward(GOOD, gt)
            assert 0.0 <= r <= 1.0


# ─── format_reward ────────────────────────────────────────────────────────────

class TestFormatReward:
    def test_has_tool_call_returns_one(self):
        assert format_reward(GOOD) == 1.0

    def test_partial_tool_call_returns_one(self):
        assert format_reward(PARTIAL_TOOL_CALL) == 1.0

    def test_no_tool_call_returns_zero(self):
        assert format_reward(NO_TOOL) == 0.0

    def test_empty_returns_zero(self):
        assert format_reward("") == 0.0

    def test_none_returns_zero(self):
        assert format_reward(None) == 0.0


# ─── combined_reward ──────────────────────────────────────────────────────────

class TestCombinedReward:
    def test_perfect_scenario(self):
        r = combined_reward(GOOD, "391", SUCCESS_SANDBOX, RewardWeights())
        assert r > 0.9

    def test_no_tool_no_answer(self):
        r = combined_reward("just thinking...", "391", None, RewardWeights())
        assert r == 0.0

    def test_clipped_to_one(self):
        r = combined_reward(GOOD, "391", SUCCESS_SANDBOX, RewardWeights())
        assert r <= 1.0

    def test_clipped_to_zero(self):
        r = combined_reward("", "", None, RewardWeights())
        assert r >= 0.0

    def test_default_weights_used_when_none(self):
        r = combined_reward(GOOD, "391", SUCCESS_SANDBOX, None)
        assert 0.0 <= r <= 1.0

    def test_custom_weights(self):
        w = RewardWeights(tool_execution=0.8, answer_correctness=0.1, format=0.1)
        r = combined_reward(GOOD, "wrong_answer", SUCCESS_SANDBOX, w)
        # High tool weight, correct execution — should still be decent
        assert r >= 0.4


# ─── RewardWeights ────────────────────────────────────────────────────────────

class TestRewardWeights:
    def test_default_weights_sum_to_one(self):
        w = RewardWeights()
        assert abs(w.tool_execution + w.answer_correctness + w.format - 1.0) < 1e-9

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            RewardWeights(tool_execution=0.9, answer_correctness=0.9, format=0.1)
