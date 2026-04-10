"""
Gradio demo for Genesis Manthan.

By default this launches a single-model demo that is suitable for use as a
Hugging Face Space app.py. Enable the baseline comparison explicitly when you
want a local side-by-side demo.

Run: python src/inference/demo.py
    python src/inference/demo.py --smoke-test
    python src/inference/demo.py --demo-mode
    python src/inference/demo.py --compare-baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MANTHAN_SYSTEM = (
    "You are Genesis Manthan, an AI agent. "
    "Solve every problem by calling tools. Never reason verbally — think through actions."
)
BASELINE_SYSTEM = (
    "You are a helpful assistant. Think step by step and answer the question."
)
DEFAULT_MODEL_ID = "Shahansha/Manthan-1.5B"
DEFAULT_BASELINE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEMO_CSS = """
.tool-call { background: #f0f7ff; border-left: 3px solid #2563eb; padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 13px; }
.tool-response { background: #f0fdf4; border-left: 3px solid #16a34a; padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 13px; }
.final-answer { background: #fefce8; border-left: 3px solid #ca8a04; padding: 10px 14px; margin: 6px 0; border-radius: 4px; font-weight: bold; }
pre { white-space: pre-wrap; word-break: break-word; margin: 4px 0; }
"""

EXAMPLE_PROBLEMS = [
    "What is the sum of all prime numbers below 100?",
    "A store has 240 items. They sell 35% on Monday and 25% of the remainder on Tuesday. How many items are left?",
    "Write a Python function to check if a number is a palindrome, then test it on 12321.",
    "What is the compound interest on $5000 at 7% per year for 10 years?",
    "Find the largest prime factor of 600851475143.",
]


def _format_manthan_output(raw: str) -> str:
    """Wrap tool calls and responses in readable HTML for the Gradio display."""
    import re
    result = raw
    result = re.sub(
        r"<tool_call>(.*?)</tool_call>",
        r'<div class="tool-call">🔧 <b>Tool Call</b><pre>\1</pre></div>',
        result, flags=re.DOTALL,
    )
    result = re.sub(
        r"<tool_response>(.*?)</tool_response>",
        r'<div class="tool-response">📤 <b>Result</b><pre>\1</pre></div>',
        result, flags=re.DOTALL,
    )
    result = re.sub(
        r"<final_answer>(.*?)</final_answer>",
        r'<div class="final-answer">✅ <b>Answer</b>: \1</div>',
        result, flags=re.DOTALL,
    )
    return result


def _build_header_markdown(model_id: str, compare_baseline: bool) -> str:
    if compare_baseline:
        comparison_copy = (
            "**Left**: Manthan-1.5B reasons through tool calls. "
            "**Right**: the base Qwen2.5-1.5B-Instruct model responds without tools."
        )
    else:
        comparison_copy = (
            "This Space runs the published Manthan checkpoint directly from the "
            "Hugging Face Hub and renders tool traces in a readable format."
        )

    return f"""
    # Genesis Manthan - Tool-Mediated Reasoning Demo
    **Model**: [{model_id}](https://huggingface.co/{model_id})

    {comparison_copy}

    Use the examples below or enter your own math, code, or reasoning task.
    """


def _generate_manthan_response(manthan_model, tokenizer, problem: str, max_new_tokens: int) -> str:
    if manthan_model is None or tokenizer is None:
        return (
            f'<tool_call>{{"name": "python_repl", "arguments": {{"code": "# Solving: {problem[:40]}...\\nprint(42)"}}}}</tool_call>'
            f'<tool_response>{{"result": "42", "success": true}}</tool_response>'
            f'<final_answer>42 (demo mode - load {DEFAULT_MODEL_ID} for a real answer)</final_answer>'
        )

    from src.inference.budget_forcing import generate_with_budget_forcing

    return generate_with_budget_forcing(
        manthan_model,
        tokenizer,
        problem,
        min_calls=1,
        max_calls=5,
        max_new_tokens=max_new_tokens,
    )


def _generate_baseline_response(baseline_model, tokenizer, problem: str, max_new_tokens: int) -> str:
    if baseline_model is None or tokenizer is None:
        return (
            "Demo mode - baseline model not loaded.\n\n"
            "The baseline model would produce a plain-text answer here without tool calls."
        )

    import torch

    messages = [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": problem},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(baseline_model.device)
    with torch.no_grad():
        out = baseline_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def build_demo(
    manthan_model=None,
    baseline_model=None,
    tokenizer=None,
    model_id: str = DEFAULT_MODEL_ID,
    compare_baseline: bool = False,
    max_new_tokens: int = 384,
):
    """Build the Gradio Blocks interface."""
    import gradio as gr

    with gr.Blocks(title="Genesis Manthan Demo") as demo:
        gr.Markdown(_build_header_markdown(model_id, compare_baseline))

        with gr.Row():
            problem_input = gr.Textbox(
                label="Problem",
                placeholder="Enter a math, coding, or factual question...",
                lines=2,
                scale=4,
            )
            with gr.Column(scale=1):
                run_btn = gr.Button("▶ Run", variant="primary")
                clear_btn = gr.Button("Clear")

        gr.Examples(examples=EXAMPLE_PROBLEMS, inputs=problem_input, label="Example Problems")

        if compare_baseline:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Manthan-1.5B")
                    manthan_out = gr.HTML(label="Formatted Tool Trace")
                    manthan_time = gr.Textbox(label="Time", interactive=False, max_lines=1)

                with gr.Column():
                    gr.Markdown("### Qwen2.5-1.5B-Instruct Baseline")
                    baseline_out = gr.Textbox(label="Baseline Output", lines=12, interactive=False)
                    baseline_time = gr.Textbox(label="Time", interactive=False, max_lines=1)

            def run_comparison(problem: str):
                if not problem.strip():
                    return "", "", "", ""

                t0 = time.time()
                manthan_raw = _generate_manthan_response(manthan_model, tokenizer, problem, max_new_tokens)
                manthan_elapsed = f"{time.time() - t0:.2f}s"

                t1 = time.time()
                baseline_raw = _generate_baseline_response(
                    baseline_model,
                    tokenizer,
                    problem,
                    max_new_tokens,
                )
                baseline_elapsed = f"{time.time() - t1:.2f}s"

                return (
                    _format_manthan_output(manthan_raw),
                    manthan_elapsed,
                    baseline_raw,
                    baseline_elapsed,
                )

            run_btn.click(
                run_comparison,
                inputs=[problem_input],
                outputs=[manthan_out, manthan_time, baseline_out, baseline_time],
            )
            clear_btn.click(
                lambda: ("", "", "", "", ""),
                outputs=[problem_input, manthan_out, manthan_time, baseline_out, baseline_time],
            )
        else:
            gr.Markdown("### Manthan-1.5B")
            manthan_out = gr.HTML(label="Formatted Tool Trace")
            manthan_time = gr.Textbox(label="Time", interactive=False, max_lines=1)
            manthan_raw_out = gr.Textbox(label="Raw Model Output", lines=10, interactive=False)

            def run_single(problem: str):
                if not problem.strip():
                    return "", "", ""

                t0 = time.time()
                manthan_raw = _generate_manthan_response(manthan_model, tokenizer, problem, max_new_tokens)
                manthan_elapsed = f"{time.time() - t0:.2f}s"
                return _format_manthan_output(manthan_raw), manthan_elapsed, manthan_raw

            run_btn.click(
                run_single,
                inputs=[problem_input],
                outputs=[manthan_out, manthan_time, manthan_raw_out],
            )
            clear_btn.click(
                lambda: ("", "", "", ""),
                outputs=[problem_input, manthan_out, manthan_time, manthan_raw_out],
            )

    return demo


def _run_smoke_test() -> None:
    print("Running demo smoke test...")
    try:
        import gradio as gr
        print(f"  OK  gradio {gr.__version__} available")
    except ImportError:
        print("  WARN gradio not installed (pip install gradio)")
        print("  OK  smoke test passed (graceful degradation)")
        return

    demo = build_demo(
        manthan_model=None,
        baseline_model=None,
        tokenizer=None,
        model_id=DEFAULT_MODEL_ID,
        compare_baseline=False,
        max_new_tokens=256,
    )
    print("  OK  Gradio Blocks built successfully")
    print("\ndemo smoke test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan — Gradio Demo")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--model", type=str, default=None, help="Model path or HF repo ID")
    parser.add_argument("--baseline-model", type=str, default=None, help="Optional baseline model ID")
    parser.add_argument("--compare-baseline", action="store_true", help="Load baseline model for side-by-side comparison")
    parser.add_argument("--demo-mode", action="store_true", help="Run without loading models")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        sys.exit(0)

    manthan_model = None
    baseline_model = None
    tokenizer = None
    model_id = args.model or os.environ.get("MANTHAN_MODEL_PATH") or DEFAULT_MODEL_ID
    baseline_model_id = (
        args.baseline_model
        or os.environ.get("MANTHAN_BASELINE_MODEL")
        or DEFAULT_BASELINE_MODEL_ID
    )
    compare_baseline = args.compare_baseline or os.environ.get("MANTHAN_ENABLE_BASELINE", "").lower() in {
        "1",
        "true",
        "yes",
    }

    if not args.demo_mode:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        hf_token = os.environ.get("HF_TOKEN")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading Manthan model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        manthan_model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype, device_map="auto", token=hf_token
        )
        manthan_model.eval()
        if hasattr(manthan_model, "generation_config") and manthan_model.generation_config is not None:
            manthan_model.generation_config.max_length = None
        if compare_baseline:
            print(f"Loading baseline: {baseline_model_id}")
            baseline_model = AutoModelForCausalLM.from_pretrained(
                baseline_model_id,
                dtype=torch_dtype,
                device_map="auto",
                token=hf_token,
            )
            baseline_model.eval()
            if hasattr(baseline_model, "generation_config") and baseline_model.generation_config is not None:
                baseline_model.generation_config.max_length = None

    demo = build_demo(
        manthan_model,
        baseline_model,
        tokenizer,
        model_id=model_id,
        compare_baseline=compare_baseline,
        max_new_tokens=args.max_new_tokens,
    )
    import gradio as gr

    demo.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=DEMO_CSS,
    )
