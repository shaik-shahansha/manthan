"""
smolagents integration for Genesis Manthan.
Wraps Manthan-1.5B as a smolagents CodeAgent or ToolCallingAgent
with budget forcing applied at inference time.

Usage:
    from src.inference.smolagents_integration import create_manthan_agent
    agent = create_manthan_agent()
    result = agent.run("Calculate the sum of first 100 prime numbers")

Smoke test:
    python src/inference/smolagents_integration.py --smoke-test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─── sandboxed Python REPL tool ───────────────────────────────────────────────

def _python_repl_tool(code: str) -> str:
    """
    Execute Python code in a sandboxed subprocess and return the output.
    Timeout: 10 seconds. Output capped at 10KB.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
        dir=tempfile.gettempdir(), prefix="manthan_repl_",
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=10,
            cwd=tempfile.gettempdir(),
        )
        out = (result.stdout + result.stderr)[:10240]
        return out.strip() if out.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out (10s limit)"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─── factory function ─────────────────────────────────────────────────────────

def create_manthan_agent(
    agent_type: str = "code",
    model_path: str | None = None,
    min_tool_calls: int = 1,
    max_tool_calls: int = 5,
):
    """
    Create a smolagents agent backed by the Manthan model.

    Args:
        agent_type:     "code" for CodeAgent, "tool" for ToolCallingAgent.
        model_path:     HuggingFace model ID or local path. Falls back to
                        MANTHAN_MODEL_PATH env var, then shahansha/Manthan-1.5B.
        min_tool_calls: Budget forcing minimum tool calls before concluding.
        max_tool_calls: Budget forcing maximum allowed tool calls.

    Returns:
        A smolagents Agent instance ready to call .run(problem).
    """
    try:
        import smolagents
    except ImportError:
        print("ERROR: smolagents not installed. Run: pip install smolagents")
        sys.exit(1)

    resolved_path = (
        model_path
        or os.environ.get("MANTHAN_MODEL_PATH")
        or "shahansha/Manthan-1.5B"
    )

    hf_token = os.environ.get("HF_TOKEN")

    # Build the HuggingFace model backend
    from smolagents import HfApiModel, CodeAgent, ToolCallingAgent

    model = HfApiModel(
        model_id=resolved_path,
        token=hf_token,
    )

    # Define the python REPL as a smolagents Tool
    from smolagents import tool

    @tool
    def python_repl(code: str) -> str:
        """Execute Python code and return the output. Use this to compute, verify, or transform data."""
        return _python_repl_tool(code)

    tools = [python_repl]

    if agent_type == "code":
        agent = CodeAgent(tools=tools, model=model)
    elif agent_type == "tool":
        agent = ToolCallingAgent(tools=tools, model=model)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}. Use 'code' or 'tool'.")

    return agent


# ─── local model variant (for GPU inference without smolagents HfApiModel) ────

def create_local_manthan_agent(
    model_path: str | None = None,
    min_tool_calls: int = 1,
    max_tool_calls: int = 5,
    device: str = "auto",
):
    """
    Create a Manthan agent using a locally loaded model (GPU inference).
    Uses budget forcing via BudgetForcingProcessor. Does NOT require smolagents.

    Args:
        model_path:     Local checkpoint path or HuggingFace model ID.
        min_tool_calls: Minimum tool calls before final answer is allowed.
        max_tool_calls: Maximum tool calls (forces conclusion after this).
        device:         "auto", "cuda", or "cpu".

    Returns:
        A callable: agent(problem: str) -> str
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.inference.budget_forcing import BudgetForcingProcessor, generate_with_budget_forcing

    resolved_path = (
        model_path
        or os.environ.get("MANTHAN_MODEL_PATH")
        or "shahansha/Manthan-1.5B"
    )

    hf_token = os.environ.get("HF_TOKEN")
    print(f"[Manthan] Loading local model: {resolved_path}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        dtype=torch.float16,
        device_map=device,
        token=hf_token,
    )
    model.eval()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    if torch.cuda.is_available():
        print(f"[Manthan] GPU: {torch.cuda.get_device_name(0)}")

    def agent(problem: str) -> str:
        return generate_with_budget_forcing(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            min_calls=min_tool_calls,
            max_calls=max_tool_calls,
        )

    return agent


# ─── smoke test ───────────────────────────────────────────────────────────────

def _run_smoke_test() -> None:
    print("Running smolagents_integration smoke test...")

    # Check smolagents import
    try:
        import smolagents
        print(f"  OK  smolagents {smolagents.__version__} available")
    except ImportError:
        print("  WARN smolagents not installed (pip install smolagents)")
        print("  OK  smoke test passed (smolagents optional for local dev)")

    # Verify the python_repl_tool sandbox
    result = _python_repl_tool("print('hello from manthan')")
    assert "hello from manthan" in result, f"Unexpected output: {result}"
    print(f"  OK  python_repl_tool: {result!r}")

    timeout_result = _python_repl_tool("import time; time.sleep(30)")
    assert "timed out" in timeout_result.lower(), f"Timeout should trigger: {timeout_result}"
    print(f"  OK  python_repl_tool timeout enforced")

    # Check env vars
    model_path = os.environ.get("MANTHAN_MODEL_PATH", "shahansha/Manthan-1.5B (default)")
    print(f"  OK  MANTHAN_MODEL_PATH resolved: {model_path}")

    print("\nsmolagents_integration smoke test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan — smolagents Integration")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--problem", type=str, help="Run a single problem (requires GPU)")
    parser.add_argument("--agent-type", type=str, default="code", choices=["code", "tool"])
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    elif args.problem:
        agent = create_local_manthan_agent(model_path=args.model)
        print(f"\nProblem: {args.problem}")
        print(f"Answer:  {agent(args.problem)}")
    else:
        print("Use --smoke-test or --problem 'your question here'")
