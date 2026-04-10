"""Opt-in integration test for generating a real response from the Hugging Face Hub."""

from __future__ import annotations

import os

import pytest

from src.inference.budget_forcing import _build_prompt


DEFAULT_MODEL_ID = "Shahansha/Manthan-1.5B"
DEFAULT_PROBLEM = "What is 144 + 256?"


def _should_run_hub_test() -> bool:
    return os.environ.get("RUN_HF_MODEL_TEST", "").strip().lower() in {"1", "true", "yes"}


@pytest.fixture(scope="module")
def hub_model_bundle() -> tuple[object, object]:
    """Load the tokenizer and model directly from the Hugging Face Hub."""
    if not _should_run_hub_test():
        pytest.skip("Set RUN_HF_MODEL_TEST=1 to run the Hugging Face model integration test.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("HF_TEST_MODEL_ID", DEFAULT_MODEL_ID)
    hf_token = os.environ.get("HF_TOKEN") or None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device_map is None:
        model.to("cpu")

    return tokenizer, model


@pytest.mark.integration
@pytest.mark.huggingface
def test_manthan_generates_structured_response_from_hub(
    hub_model_bundle: tuple[object, object],
) -> None:
    import torch

    tokenizer, model = hub_model_bundle
    problem = os.environ.get("HF_TEST_PROMPT", DEFAULT_PROBLEM)
    max_new_tokens = int(os.environ.get("HF_TEST_MAX_NEW_TOKENS", "128"))

    prompt = _build_prompt(tokenizer, problem)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)

    assert response.strip(), "Expected a non-empty response from the Hub model."
    assert any(marker in response for marker in ("<tool_call>", "<final_answer>")), (
        "Expected Manthan to emit a structured tool-use response, but got:\n"
        f"{response}"
    )

    if "<final_answer>" in response:
        assert "400" in response, f"Expected the arithmetic answer to appear in the final output, got:\n{response}"