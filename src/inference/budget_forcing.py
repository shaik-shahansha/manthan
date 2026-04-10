"""
Budget Forcing LogitsProcessor for Genesis Manthan.

Forces the model to make additional tool calls before producing <final_answer>,
and caps total tool calls at a maximum budget.

Based on: arXiv:2510.21398 (budget forcing for 1.5B models)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_TOOL_CALL_RE = re.compile(r"<tool_call>")
_FINAL_ANSWER_STR = "<final_answer>"


class BudgetForcingProcessor:
    """
    LogitsProcessor that enforces a minimum and maximum number of tool calls.

    During generation:
    - If the model is about to emit the <final_answer> token but has used fewer
      than minimum_tool_calls tool calls, suppress that token and boost "Wait".
    - If the model has used maximum_tool_calls or more, boost <final_answer>
      to force the model to conclude.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        minimum_tool_calls: int = 1,
        maximum_tool_calls: int = 5,
    ) -> None:
        """
        Args:
            tokenizer: The model's tokenizer, used to look up token IDs.
            minimum_tool_calls: Model must make at least this many tool calls.
            maximum_tool_calls: Model is forced to conclude after this many.
        """
        self.tokenizer = tokenizer
        self.minimum_tool_calls = minimum_tool_calls
        self.maximum_tool_calls = maximum_tool_calls

        # Look up token IDs — never hardcode
        self._final_answer_ids: list[int] = self._find_token_ids(_FINAL_ANSWER_STR)
        self._wait_ids: list[int] = self._find_token_ids("Wait")
        self._tool_call_str = "<tool_call>"

    def _find_token_ids(self, text: str) -> list[int]:
        """Return token IDs for the given string fragment."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids

    def _count_tool_calls(self, decoded: str) -> int:
        """Count how many <tool_call> blocks have been opened so far."""
        return len(_TOOL_CALL_RE.findall(decoded))

    def _is_generating_final_answer(self, scores_row, threshold: float = 2.0) -> bool:
        """
        Check if the top token is likely the start of <final_answer>.

        We look at whether the first token of _final_answer_ids has the highest
        (or near-highest) score.
        """
        if not self._final_answer_ids:
            return False
        import torch

        first_id = self._final_answer_ids[0]
        top_token = int(torch.argmax(scores_row).item())
        return top_token == first_id

    def __call__(self, input_ids, scores):
        """
        Apply budget forcing to the logit scores.

        Args:
            input_ids: LongTensor of shape (batch, seq_len).
            scores: FloatTensor of shape (batch, vocab_size).

        Returns:
            Modified scores FloatTensor.
        """
        import torch

        modified = scores.clone()

        for b in range(input_ids.shape[0]):
            # Decode the current generation for this batch item
            decoded = self.tokenizer.decode(input_ids[b], skip_special_tokens=False)
            tool_call_count = self._count_tool_calls(decoded)

            is_about_to_conclude = self._is_generating_final_answer(modified[b])

            if tool_call_count >= self.maximum_tool_calls:
                # Force the model to conclude — boost <final_answer> first token
                if self._final_answer_ids:
                    modified[b, self._final_answer_ids[0]] += 10.0

            elif is_about_to_conclude and tool_call_count < self.minimum_tool_calls:
                # Suppress <final_answer>, inject "Wait"
                if self._final_answer_ids:
                    modified[b, self._final_answer_ids[0]] = float("-inf")
                if self._wait_ids:
                    modified[b, self._wait_ids[0]] += 5.0

        return modified


def generate_with_budget_forcing(
    model,
    tokenizer: "PreTrainedTokenizer",
    problem: str,
    min_calls: int = 1,
    max_calls: int = 5,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate a response with budget forcing enabled.

    Args:
        model: The loaded language model.
        tokenizer: The model's tokenizer.
        problem: The user's problem string.
        min_calls: Minimum tool calls before concluding.
        max_calls: Maximum tool calls allowed.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The generated text string.
    """
    from transformers import LogitsProcessorList  # type: ignore[import]

    processor = BudgetForcingProcessor(tokenizer, min_calls, max_calls)
    prompt = _build_prompt(tokenizer, problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([processor]),
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def _build_prompt(tokenizer: "PreTrainedTokenizer", problem: str) -> str:
    """Build the ChatML prompt used for Manthan generation."""
    system = (
        "You are Genesis Manthan, an AI agent that solves problems by calling tools. "
        "Never reason verbally — always reason through tool execution."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _run_smoke_test() -> None:
    print("Running budget_forcing smoke test...\n")

    import torch
    from transformers import AutoTokenizer  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print("  Tokenizer loaded ✓")

    processor = BudgetForcingProcessor(tokenizer, minimum_tool_calls=1, maximum_tool_calls=3)
    print(f"  BudgetForcingProcessor created ✓")
    print(f"  <final_answer> token IDs: {processor._final_answer_ids}")
    print(f"  'Wait' token IDs:         {processor._wait_ids}")

    # Create mock input: input_ids and scores on CPU
    vocab_size = tokenizer.vocab_size or 32000
    batch_size = 2
    seq_len = 10

    mock_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    mock_scores = torch.randn((batch_size, vocab_size))

    # Run forward pass
    modified_scores = processor(mock_input_ids, mock_scores)

    assert modified_scores.shape == mock_scores.shape, (
        f"Score shape mismatch: {modified_scores.shape} != {mock_scores.shape}"
    )
    print(f"  Forward pass shape preserved ({batch_size}, {vocab_size}) ✓")

    # Verify no NaN
    assert not torch.isnan(modified_scores).any(), "NaN values in modified scores"
    print("  No NaN in output scores ✓")

    print("\nbudget_forcing smoke test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan budget forcing")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests and exit")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        sys.exit(0)

    print("budget_forcing.py — import BudgetForcingProcessor or use --smoke-test")
