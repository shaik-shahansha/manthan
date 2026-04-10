"""Unit tests for GRPO rollout helpers."""

from __future__ import annotations

from src.training.grpo_train import (
    _build_rollout_prompt,
    _build_tool_response_block,
    _completion_to_text,
    _extract_first_tool_call_payload,
    _prompt_to_chatml,
)


def test_completion_to_text_normalizes_chat_messages() -> None:
    completion = [
        {"role": "assistant", "content": '<tool_call>{"name":"python_repl"}</tool_call>'},
        {"role": "assistant", "content": "<final_answer>4</final_answer>"},
    ]

    assert _completion_to_text(completion) == (
        '<tool_call>{"name":"python_repl"}</tool_call>\n<final_answer>4</final_answer>'
    )


def test_prompt_to_chatml_preserves_tool_role_format() -> None:
    prompt = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    assert _prompt_to_chatml(prompt) == (
        "<|im_start|>system\nUse tools.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>"
    )


def test_extract_first_tool_call_payload_handles_partial_block() -> None:
    completion = (
        '<tool_call>{"name":"python_repl","arguments":{"code":"print(2+2)"}}\n'
        '<tool_response>{"result":"4","success":true}</tool_response>'
    )

    assert _extract_first_tool_call_payload(completion) == (
        '{"name":"python_repl","arguments":{"code":"print(2+2)"}}'
    )


def test_build_tool_response_block_includes_error_when_present() -> None:
    sandbox_result = {"success": False, "result": "", "error": "timeout"}

    assert _build_tool_response_block(sandbox_result) == (
        '<tool_response>{"result": "", "success": false, "error": "timeout"}</tool_response>'
    )


def test_build_rollout_prompt_adds_tool_observation_turn() -> None:
    prompt = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    tool_call_block = '<tool_call>{"name":"python_repl","arguments":{"code":"print(2+2)"}}</tool_call>'
    sandbox_result = {"success": True, "result": "4", "error": ""}

    rollout_prompt = _build_rollout_prompt(prompt, tool_call_block, sandbox_result)

    assert "<|im_start|>assistant\n<tool_call>" in rollout_prompt
    assert "<|im_start|>tool\n<tool_response>{\"result\": \"4\", \"success\": true}</tool_response><|im_end|>" in rollout_prompt
    assert rollout_prompt.endswith("<|im_start|>assistant\n")