"""Interactive Hugging Face chat CLI for Genesis Manthan."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.format_dataset import SYSTEM_MESSAGE
from src.training.grpo_train import _build_tool_response_block, execute_code_sandbox


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|<tool_response>|<final_answer>|$)", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)


def _configure_utf8_output() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def load_model_and_tokenizer(
    model_id: str,
    device: str,
) -> tuple[Any, Any]:
    """Load a tokenizer and causal LM from the Hugging Face Hub."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN") or None
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    torch_dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    model.generation_config.max_length = None

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not use_cuda:
        model.to("cpu")

    return tokenizer, model


def build_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Build a chat prompt with the model tokenizer's chat template."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate a single assistant response for the current chat history."""
    import torch

    prompt = build_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def _extract_first_tool_call_payload(text: str) -> str | None:
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _normalise_assistant_tool_call(payload: str) -> str:
    return f"<tool_call>{payload}</tool_call>"


def _run_tool_call(payload: str, timeout_seconds: int) -> tuple[str, dict[str, Any]]:
    parsed = json.loads(payload)
    tool_name = str(parsed.get("name", "")).strip()
    arguments = parsed.get("arguments", {})

    if tool_name != "python_repl":
        return tool_name or "unknown", {
            "success": False,
            "result": "",
            "error": f"Unsupported tool: {tool_name or 'unknown'}",
        }

    code = ""
    if isinstance(arguments, dict):
        code = str(arguments.get("code", ""))

    if not code.strip():
        return tool_name, {"success": False, "result": "", "error": "Empty code payload"}

    return tool_name, execute_code_sandbox(code, timeout_seconds=timeout_seconds)


def generate_with_tool_loop(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    max_tool_rounds: int,
    sandbox_timeout: int,
) -> list[dict[str, str]]:
    """Generate until final answer or tool budget is reached."""
    transcript: list[dict[str, str]] = []

    for _ in range(max_tool_rounds + 1):
        response = generate_response(model, tokenizer, messages, max_new_tokens, temperature).strip()
        if not response:
            transcript.append({"role": "assistant", "content": ""})
            return transcript

        payload = _extract_first_tool_call_payload(response)
        if not payload:
            messages.append({"role": "assistant", "content": response})
            transcript.append({"role": "assistant", "content": response})
            return transcript

        assistant_tool_call = _normalise_assistant_tool_call(payload)
        messages.append({"role": "assistant", "content": assistant_tool_call})
        transcript.append({"role": "assistant", "content": assistant_tool_call})

        try:
            tool_name, sandbox_result = _run_tool_call(payload, timeout_seconds=sandbox_timeout)
        except json.JSONDecodeError as exc:
            tool_name = "python_repl"
            sandbox_result = {"success": False, "result": "", "error": f"Invalid tool JSON: {exc}"}

        tool_response = _build_tool_response_block(sandbox_result)
        messages.append({"role": "tool", "content": tool_response})
        transcript.append({"role": tool_name, "content": tool_response})

        if "<final_answer>" in response:
            return transcript

    fallback = "<final_answer>Stopped after reaching the maximum tool-call budget.</final_answer>"
    messages.append({"role": "assistant", "content": fallback})
    transcript.append({"role": "assistant", "content": fallback})
    return transcript


def _print_transcript(transcript: list[dict[str, str]]) -> None:
    for item in transcript:
        role = item["role"]
        content = item["content"].replace("<|im_end|>", "").strip()
        if role == "assistant" and content.startswith("<tool_call>"):
            payload = _extract_first_tool_call_payload(content) or ""
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                tool_name = parsed.get("name", "unknown")
                code = ""
                arguments = parsed.get("arguments", {})
                if isinstance(arguments, dict):
                    code = str(arguments.get("code", "")).rstrip()
                print(f"\nTool Call> {tool_name}")
                if code:
                    print(code)
            else:
                print(f"\nTool Call> {content}")
        elif role == "python_repl":
            match = _TOOL_RESPONSE_RE.search(content)
            if match:
                try:
                    payload = json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    success = payload.get("success", False)
                    result = str(payload.get("result", "")).strip()
                    error = str(payload.get("error", "")).strip()
                    print(f"\nTool Result> success={success}")
                    if result:
                        print(result)
                    if error:
                        print(f"error: {error}")
                else:
                    print(f"\nTool Result> {content}")
            else:
                print(f"\nTool Result> {content}")
        elif role == "assistant":
            final_answer = _FINAL_ANSWER_RE.search(content)
            if final_answer:
                print(f"\nFinal Answer> {final_answer.group(1).strip()}")
            else:
                print(f"\nModel> {content}")
        else:
            print(f"\n{role}> {content}")


def run_smoke_test(model_id: str) -> None:
    """Run a CPU-safe smoke test that loads only the tokenizer."""
    from transformers import AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    prompt = build_prompt(messages, tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")

    print("Smoke test passed")
    print(f"Model: {model_id}")
    print(f"Prompt tokens: {encoded['input_ids'].shape[1]}")


def run_repl(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    max_tool_rounds: int,
    sandbox_timeout: int,
) -> None:
    """Run an interactive question-answer loop against the loaded model."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("Interactive Manthan chat")
    print("Type a question and press Enter.")
    print("Commands: /reset, /system, /exit")

    while True:
        user_text = input("\nYou> ").strip()

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if user_text.lower() == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("Conversation reset.")
            continue
        if user_text.lower() == "/system":
            print(f"System prompt:\n{system_prompt}")
            continue

        messages.append({"role": "user", "content": user_text})
        transcript = generate_with_tool_loop(
            model,
            tokenizer,
            messages,
            max_new_tokens,
            temperature,
            max_tool_rounds,
            sandbox_timeout,
        )
        _print_transcript(transcript)


def main() -> None:
    """Parse arguments and run smoke test, one-shot generation, or interactive chat."""
    _configure_utf8_output()

    parser = argparse.ArgumentParser(description="Interactive Hugging Face chat for Genesis Manthan")
    parser.add_argument("--model", default="Shahansha/Manthan-1.5B", help="Hub model ID")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--system-prompt", default=SYSTEM_MESSAGE)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    parser.add_argument("--sandbox-timeout", type=int, default=10)
    parser.add_argument("--prompt", help="Run one prompt and exit")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but no GPU is available.")

    if args.smoke_test:
        run_smoke_test(args.model)
        return

    tokenizer, model = load_model_and_tokenizer(args.model, args.device)

    if args.prompt:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        transcript = generate_with_tool_loop(
            model,
            tokenizer,
            messages,
            args.max_new_tokens,
            args.temperature,
            args.max_tool_rounds,
            args.sandbox_timeout,
        )
        _print_transcript(transcript)
        return

    run_repl(
        model,
        tokenizer,
        args.system_prompt,
        args.max_new_tokens,
        args.temperature,
        args.max_tool_rounds,
        args.sandbox_timeout,
    )


if __name__ == "__main__":
    main()