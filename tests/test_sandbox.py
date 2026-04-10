"""Unit tests for the sandbox (used in GRPO and eval) — CPU-only."""

from __future__ import annotations

import pytest
from src.training.grpo_train import execute_code_sandbox


class TestSandbox:
    def test_simple_arithmetic(self):
        r = execute_code_sandbox("print(2 + 2)")
        assert r["success"] is True
        assert r["result"] == "4"

    def test_multiline_code(self):
        code = "x = 10\ny = 20\nprint(x + y)"
        r = execute_code_sandbox(code)
        assert r["success"] is True
        assert r["result"] == "30"

    def test_syntax_error_caught(self):
        r = execute_code_sandbox("def broken(:\n    pass")
        assert r["success"] is False
        assert r["error"]

    def test_runtime_error_caught(self):
        r = execute_code_sandbox("x = 1 / 0")
        assert r["success"] is False

    def test_timeout_enforced(self):
        r = execute_code_sandbox("import time; time.sleep(30)", timeout_seconds=2)
        assert r["success"] is False
        assert "timeout" in r["error"].lower()

    def test_import_error_caught(self):
        r = execute_code_sandbox("import module_that_does_not_exist_xyz")
        assert r["success"] is False

    def test_output_captured(self):
        r = execute_code_sandbox("for i in range(5): print(i)")
        assert r["success"] is True
        assert "4" in r["result"]

    def test_empty_output_handled(self):
        r = execute_code_sandbox("x = 42  # no print")
        assert r["success"] is True
        assert r["result"] == ""
