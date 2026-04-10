"""
Synthetic tool-interaction trace generator for Genesis Manthan.

Supports Google Gemini (primary), Anthropic Claude, and OpenAI.
API keys are loaded from .env automatically.

Usage:
  Smoke test (no API call):  python src/data/generate_synthetic.py --smoke-test
  Dummy data (no API key):   python src/data/generate_synthetic.py --dummy-data --n-samples 20
  Generate with Gemini:      python src/data/generate_synthetic.py --provider gemini --n-samples 200
  Generate with Claude:      python src/data/generate_synthetic.py --provider anthropic --n-samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Load .env automatically (safe to call even if .env doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables directly

# ---------------------------------------------------------------------------
# Seed problems — 100 total, 25 per domain, fully hardcoded
# ---------------------------------------------------------------------------

SEED_PROBLEMS: list[dict] = [
    # --- math (25) ---
    {"domain": "math", "problem": "A store had 240 items. They sold 35% on Monday and 25% of the remainder on Tuesday. How many items are left?"},
    {"domain": "math", "problem": "A train travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. What is the total distance?"},
    {"domain": "math", "problem": "Lisa earns $1,200/month. She saves 25% and spends $300 on rent. How much is left after savings and rent?"},
    {"domain": "math", "problem": "A rectangle has perimeter 56 cm. Its length is 3 times its width. Find the area."},
    {"domain": "math", "problem": "If 8 workers can build a wall in 12 days, how many days will 6 workers take?"},
    {"domain": "math", "problem": "A shopkeeper buys an item for $400 and sells it with a 30% profit. What is the selling price?"},
    {"domain": "math", "problem": "How many seconds are there in 3 weeks?"},
    {"domain": "math", "problem": "A tank is 40% full with 200 litres. What is the full capacity?"},
    {"domain": "math", "problem": "John scored 78, 85, 92, 67, and 88 in five tests. What is his average score?"},
    {"domain": "math", "problem": "A car depreciates by 15% per year. If it costs $20,000 now, what will it be worth in 3 years?"},
    {"domain": "math", "problem": "Two numbers sum to 84. The larger is 3 times the smaller. Find both numbers."},
    {"domain": "math", "problem": "A rope is 15.6 metres long. It is cut into pieces of 0.6 m each. How many pieces are there?"},
    {"domain": "math", "problem": "What is the compound interest on $5000 at 8% per year for 2 years?"},
    {"domain": "math", "problem": "A field is 120m × 80m. How many full laps of the perimeter equal at least 1 km?"},
    {"domain": "math", "problem": "The sum of first N natural numbers is 210. Find N."},
    {"domain": "math", "problem": "A recipe for 4 people needs 300g flour. How much flour for 7 people?"},
    {"domain": "math", "problem": "Convert 98.6°F to Celsius. (Formula: C = (F - 32) × 5/9)"},
    {"domain": "math", "problem": "A sphere has radius 7 cm. Find its volume to 2 decimal places. (V = 4/3 × π × r³)"},
    {"domain": "math", "problem": "17 is what percentage of 340?"},
    {"domain": "math", "problem": "A cistern can be filled in 6 hours and emptied in 8 hours. If both pipes open together, how long to fill it?"},
    {"domain": "math", "problem": "If log₂(x) = 5, what is x?"},
    {"domain": "math", "problem": "Find the number of prime numbers between 1 and 50."},
    {"domain": "math", "problem": "A bag has 5 red, 3 blue, and 2 green balls. What is the probability of picking a blue ball?"},
    {"domain": "math", "problem": "What is 2^10 + 2^8?"},
    {"domain": "math", "problem": "The hypotenuse of a right triangle is 13 cm and one leg is 5 cm. Find the other leg."},

    # --- code_debug (25) ---
    {"domain": "code_debug", "problem": "Fix this function: def sum_list(lst):\n    total = 0\n    for i in range(1, len(lst)):\n        total += lst[i]\n    return total\n# Bug: should sum ALL elements"},
    {"domain": "code_debug", "problem": "Fix this: def factorial(n):\n    if n == 0: return 0\n    return n * factorial(n-1)\n# Bug: factorial(0) should return 1"},
    {"domain": "code_debug", "problem": "Fix this: def is_palindrome(s):\n    return s == s.reverse()\n# Bug: str has no .reverse() method"},
    {"domain": "code_debug", "problem": "Fix this: def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 23\n# Bug: should add 32 not 23"},
    {"domain": "code_debug", "problem": "Fix this: def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n# Bug: misses uppercase vowels"},
    {"domain": "code_debug", "problem": "Fix this: def binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid\n        else: hi = mid\n    return -1\n# Bug: infinite loop when arr[mid] < target"},
    {"domain": "code_debug", "problem": "Fix this: def flatten(lst):\n    result = []\n    for item in lst:\n        if type(item) == list:\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n# Is there a bug? Test with [[1,[2,3]],[4]]"},
    {"domain": "code_debug", "problem": "Fix this FizzBuzz: for i in range(1, 101):\n    if i % 3: print('Fizz')\n    elif i % 5: print('Buzz')\n    else: print(i)\n# Bug: condition logic is inverted"},
    {"domain": "code_debug", "problem": "Fix this: def remove_duplicates(lst):\n    seen = []\n    return [x for x in lst if x not in seen and not seen.append(x)]\n# Is this correct? Verify with [1,2,1,3,2]"},
    {"domain": "code_debug", "problem": "Fix this: def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n# Test it: is gcd(48, 18) correct?"},
    {"domain": "code_debug", "problem": "Fix: words = 'hello world foo bar'\nword_count = {}\nfor w in words:\n    word_count[w] = word_count.get(w, 0) + 1\n# Bug: iterating over characters not words"},
    {"domain": "code_debug", "problem": "Fix: def safe_divide(a, b):\n    try:\n        return a / b\n    except:\n        return None\n# The bare except is bad practice — fix it"},
    {"domain": "code_debug", "problem": "Fix: matrix = [[0]*3]*3\nmatrix[0][0] = 1\nprint(matrix)\n# Bug: all rows share the same list object"},
    {"domain": "code_debug", "problem": "This function should return the first duplicate in a list. Fix it:\ndef first_duplicate(lst):\n    seen = set()\n    for x in lst:\n        if x in seen:\n            seen.add(x)\n            return x\n    return None"},
    {"domain": "code_debug", "problem": "Fix: import math\ndef circle_area(r):\n    return math.pi * r ^ 2\n# Bug: ^ is XOR in Python, not exponentiation"},
    {"domain": "code_debug", "problem": "Fix: def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) or j < len(b):\n        if a[i] < b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result\n# Bug: index out of range when one list is exhausted"},
    {"domain": "code_debug", "problem": "Fix: def chunk_list(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]\n# Is this correct? Test with [1..10], n=3"},
    {"domain": "code_debug", "problem": "Fix: text = 'Hello World 123'\ndigits = filter(str.isdigit, text)\nprint(list(digits))\n# Will this print individual digits or '123'?"},
    {"domain": "code_debug", "problem": "Fix: def running_average(nums):\n    total = 0\n    for i, n in enumerate(nums):\n        total += n\n        nums[i] = total / i\n    return nums\n# Bug: ZeroDivisionError at first element"},
    {"domain": "code_debug", "problem": "Fix: import random\nrandom.seed(42)\nresult = [random.randint(1, 10) for _ in range(5)]\nprint(sorted(result, reverse=True)[:3])\n# What does this print? Verify by running."},
    {"domain": "code_debug", "problem": "Fix: def caesar_cipher(text, shift):\n    result = ''\n    for ch in text:\n        if ch.isalpha():\n            result += chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))\n        else:\n            result += ch\n    return result\n# Bug: doesn't handle uppercase letters"},
    {"domain": "code_debug", "problem": "Fix: stack = []\nstack.append(1); stack.append(2); stack.append(3)\ntop = stack[0]  # Bug: should peek at top of stack"},
    {"domain": "code_debug", "problem": "Fix: def nth_fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return b\n# Bug: off-by-one — nth_fibonacci(1) should return 1"},
    {"domain": "code_debug", "problem": "Fix: data = {'a': 1, 'b': 2, 'c': 3}\nfor key in data:\n    if data[key] % 2 == 0:\n        del data[key]\n# Bug: RuntimeError — dict size changed during iteration"},
    {"domain": "code_debug", "problem": "Fix: def rotate_list(lst, k):\n    k = k % len(lst)\n    return lst[k:] + lst[:k]\n# Is this a left or right rotation? Verify with [1,2,3,4,5], k=2"},

    # --- factual_qa (25) ---
    {"domain": "factual_qa", "problem": "How many seconds are in a leap year?"},
    {"domain": "factual_qa", "problem": "What is the speed of light in km/s (rounded to nearest integer)?"},
    {"domain": "factual_qa", "problem": "How many bits are in 1 terabyte (using 1TB = 1024^4 bytes)?"},
    {"domain": "factual_qa", "problem": "What is the sum of all ASCII codes for the word 'Python'?"},
    {"domain": "factual_qa", "problem": "How many days are between January 1, 2024 and July 4, 2024 (inclusive)?"},
    {"domain": "factual_qa", "problem": "What is 100 factorial's last non-zero digit?"},
    {"domain": "factual_qa", "problem": "Convert 255 from decimal to binary and hexadecimal."},
    {"domain": "factual_qa", "problem": "What is the 100th prime number?"},
    {"domain": "factual_qa", "problem": "How many zeros are at the end of 50 factorial?"},
    {"domain": "factual_qa", "problem": "What is the GCD of 1071 and 462?"},
    {"domain": "factual_qa", "problem": "What is the LCM of 12, 15, and 20?"},
    {"domain": "factual_qa", "problem": "What is the area of a regular hexagon with side length 6 cm?"},
    {"domain": "factual_qa", "problem": "How many ways can 5 people sit in a row?"},
    {"domain": "factual_qa", "problem": "What is the sum of digits of 2^100?"},
    {"domain": "factual_qa", "problem": "What is e (Euler's number) to 10 decimal places?"},
    {"domain": "factual_qa", "problem": "What is the 20th Fibonacci number (0-indexed, starting F(0)=0)?"},
    {"domain": "factual_qa", "problem": "How many months have exactly 30 days?"},
    {"domain": "factual_qa", "problem": "What is 1 + 2 + 3 + ... + 1000?"},
    {"domain": "factual_qa", "problem": "What is the square root of 1764?"},
    {"domain": "factual_qa", "problem": "How many characters are in the string 'The quick brown fox jumps over the lazy dog'?"},
    {"domain": "factual_qa", "problem": "What is 17^4?"},
    {"domain": "factual_qa", "problem": "How many unique permutations are there of the letters in 'MISSISSIPPI'?"},
    {"domain": "factual_qa", "problem": "What is the result of 0b1010 XOR 0b1100 in decimal?"},
    {"domain": "factual_qa", "problem": "What is the distance between points (3, 4) and (7, 1) in 2D?"},
    {"domain": "factual_qa", "problem": "What day of the week was January 1, 2000?"},

    # --- data_analysis (25) ---
    {"domain": "data_analysis", "problem": "Given [4, 7, 2, 9, 1, 5, 8, 3, 6, 10], find the median and mean, then return all values above the mean."},
    {"domain": "data_analysis", "problem": "Given scores [78, 85, 92, 67, 88, 95, 73, 81], find the 75th percentile."},
    {"domain": "data_analysis", "problem": "Given the list [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], what is the mode?"},
    {"domain": "data_analysis", "problem": "Given [10, 20, 30, 40, 50], compute the standard deviation."},
    {"domain": "data_analysis", "problem": "Given prices [100, 105, 98, 110, 107, 95, 112], compute the 7-day simple moving average."},
    {"domain": "data_analysis", "problem": "Group this list by even/odd: [1,2,3,4,5,6,7,8,9,10]. Return counts for each group."},
    {"domain": "data_analysis", "problem": "Given {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'Diana': 95, 'Eve': 88}, sort by score descending and return top-3."},
    {"domain": "data_analysis", "problem": "Find all numbers in [1..100] that are both perfect squares and divisible by 4."},
    {"domain": "data_analysis", "problem": "Given [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], compute the cumulative sum at each position."},
    {"domain": "data_analysis", "problem": "Given the text 'To be or not to be that is the question', count word frequencies and return the top-3 most frequent words."},
    {"domain": "data_analysis", "problem": "Given [[1,2,3],[4,5,6],[7,8,9]], compute the sum of each row and each column."},
    {"domain": "data_analysis", "problem": "Given stock prices [10, 12, 11, 14, 13, 15, 16], compute daily percentage changes."},
    {"domain": "data_analysis", "problem": "Given [55, 62, 48, 71, 33, 89, 45, 60, 77, 52], find the interquartile range (IQR)."},
    {"domain": "data_analysis", "problem": "Given two lists A=[1,3,5,7,9] and B=[2,3,5,8,9], find their intersection, union, and elements only in A."},
    {"domain": "data_analysis", "problem": "Given the list of tuples [(1,'a'),(2,'b'),(3,'a'),(4,'c'),(5,'b')], group by the string and sum the numbers."},
    {"domain": "data_analysis", "problem": "Given [[1,4],[2,3],[5,1],[3,2]], sort by the second element ascending."},
    {"domain": "data_analysis", "problem": "Find all pairs from [1,2,3,4,5,6] that sum to 7."},
    {"domain": "data_analysis", "problem": "Given [1, 1, 2, 3, 5, 8, 13, 21, 34, 55], find the ratio of each element to the previous (should approach golden ratio)."},
    {"domain": "data_analysis", "problem": "Given {'Jan':120,'Feb':95,'Mar':135,'Apr':110,'May':155,'Jun':140}, plot a monthly trend: which months are above average?"},
    {"domain": "data_analysis", "problem": "Flatten this nested list: [1, [2, [3, 4], 5], [6, 7]] and sort the result."},
    {"domain": "data_analysis", "problem": "Given 'hello world', build a dictionary mapping each character to its frequency (excluding spaces)."},
    {"domain": "data_analysis", "problem": "Given [5, 3, 8, 1, 9, 2, 7, 4, 6], perform a bubble sort step-by-step and count the number of swaps."},
    {"domain": "data_analysis", "problem": "Given ages [22, 35, 28, 41, 19, 53, 33, 27, 45, 31], bucket them into decades (20s, 30s, 40s, 50s) and count each."},
    {"domain": "data_analysis", "problem": "Given a list of strings ['apple', 'Banana', 'cherry', 'Date', 'elderberry'], sort case-insensitively and return the result."},
    {"domain": "data_analysis", "problem": "Given [2, 4, 6, 8, 10], compute the product of all elements, sum of squares, and sum of square roots."},
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Configuration for synthetic trace generation."""

    output_path: Path = Path("data/raw/synthetic_traces.jsonl")
    n_samples: int = 100
    model: str = "gemini-2.5-flash"
    max_tokens: int = 1024
    temperature: float = 0.7
    max_retries: int = 5
    retry_delay: float = 5.0
    request_delay: float = 13.0  # seconds between requests; 13s ≈ 4.5 req/min (free tier: 5/min)
    provider: Literal["gemini", "anthropic", "openai"] = "gemini"
    min_tool_calls: int = 1


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are generating training data for a language model that reasons ONLY through tool execution — never through verbal chain-of-thought.

Generate a tool-interaction trace for the given problem using EXACTLY this format:

<tool_call>{"name": "python_repl", "arguments": {"code": "# your Python code here"}}</tool_call>
<tool_response>{"result": "execution output here", "success": true}</tool_response>
<final_answer>answer here</final_answer>

Rules:
1. Include 1 to 3 tool calls. Never more than 3.
2. NO verbal reasoning before tool calls. Jump straight to the first <tool_call>.
3. Each <tool_call> MUST be followed by a <tool_response> with realistic output.
4. The <final_answer> comes ONLY after at least one successful tool execution.
5. Code in tool_call must be valid Python that would actually produce the stated output.
6. tool_response must contain the actual printed output of the code.
7. For code_debug problems: show the buggy code being tested, then the fix being verified."""

USER_TEMPLATE = "Problem: {problem}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)


def validate_trace(trace: str, min_tool_calls: int = 1) -> tuple[bool, str]:
    """
    Validate that a generated trace has the required structure.

    Returns:
        Tuple of (is_valid, reason).
    """
    calls = _TOOL_CALL_RE.findall(trace)
    responses = _TOOL_RESPONSE_RE.findall(trace)
    answers = _FINAL_ANSWER_RE.findall(trace)

    if len(calls) < min_tool_calls:
        return False, f"Too few tool calls: {len(calls)} < {min_tool_calls}"
    if len(responses) < len(calls):
        return False, f"Missing tool_response blocks: {len(responses)} < {len(calls)}"
    if not answers:
        return False, "Missing <final_answer> block"

    for i, call_content in enumerate(calls):
        try:
            json.loads(call_content.strip())
        except json.JSONDecodeError as exc:
            return False, f"tool_call[{i}] is not valid JSON: {exc}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------


def _generate_with_gemini(problem: str, config: GeneratorConfig) -> str:
    """Call Google Gemini API to generate a single trace."""
    import re as _re
    from google import genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to .env or set the environment variable.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    client = genai.Client(api_key=api_key)
    model_name = config.model if config.provider == "gemini" else "gemini-2.5-flash"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=USER_TEMPLATE.format(problem=problem),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=config.max_tokens,
                temperature=config.temperature,
            ),
        )
        return response.text or ""
    except Exception as exc:
        exc_str = str(exc)
        # Parse retryDelay from 429 RESOURCE_EXHAUSTED errors
        if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
            # Extract 'retryDelay': '26s' from error message
            match = _re.search(r"'retryDelay':\s*'(\d+)s'", exc_str)
            if match:
                wait_secs = int(match.group(1)) + 2  # +2s buffer
                print(f"      Rate limit hit — waiting {wait_secs}s as specified by API...")
                time.sleep(wait_secs)
        raise


def _generate_with_anthropic(problem: str, config: GeneratorConfig) -> str:
    """Call Anthropic Claude API to generate a single trace."""
    import anthropic  # type: ignore[import]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=config.model if config.provider == "anthropic" else "claude-3-5-haiku-20241022",
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(problem=problem)}],
    )
    return message.content[0].text


def _generate_with_openai(problem: str, config: GeneratorConfig) -> str:
    """Call OpenAI API to generate a single trace."""
    import openai  # type: ignore[import]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=config.model if config.provider == "openai" else "gpt-4o-mini",
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(problem=problem)},
        ],
    )
    return response.choices[0].message.content or ""


# Pre-built dummy traces for --dummy-data mode (no API needed)
_DUMMY_TRACES: list[str] = [
    '<tool_call>{"name": "python_repl", "arguments": {"code": "print(240 * (1 - 0.35) * (1 - 0.25))"}}</tool_call>\n<tool_response>{"result": "117.0", "success": true}</tool_response>\n<final_answer>117</final_answer>',
    '<tool_call>{"name": "python_repl", "arguments": {"code": "print(60*2.5 + 80*1.5)"}}</tool_call>\n<tool_response>{"result": "270.0", "success": true}</tool_response>\n<final_answer>270 km</final_answer>',
    '<tool_call>{"name": "python_repl", "arguments": {"code": "print(1200 - 1200*0.25 - 300)"}}</tool_call>\n<tool_response>{"result": "600.0", "success": true}</tool_response>\n<final_answer>$600</final_answer>',
    '<tool_call>{"name": "python_repl", "arguments": {"code": "w=56/8; l=3*w; print(l*w)"}}</tool_call>\n<tool_response>{"result": "147.0", "success": true}</tool_response>\n<final_answer>147 sq cm</final_answer>',
    '<tool_call>{"name": "python_repl", "arguments": {"code": "print(8*12/6)"}}</tool_call>\n<tool_response>{"result": "16.0", "success": true}</tool_response>\n<final_answer>16 days</final_answer>',
]


def generate_trace(problem: str, config: GeneratorConfig) -> str | None:
    """
    Generate a single tool-interaction trace for a problem.

    Retries up to config.max_retries times if validation fails.

    Returns:
        The trace string, or None if all retries fail.
    """
    for attempt in range(1, config.max_retries + 1):
        try:
            if config.provider == "gemini":
                trace = _generate_with_gemini(problem, config)
            elif config.provider == "anthropic":
                trace = _generate_with_anthropic(problem, config)
            else:
                trace = _generate_with_openai(problem, config)

            is_valid, reason = validate_trace(trace, config.min_tool_calls)
            if is_valid:
                return trace

            print(f"    Attempt {attempt}/{config.max_retries}: invalid trace — {reason}")
            time.sleep(config.retry_delay)

        except Exception as exc:  # noqa: BLE001
            print(f"    Attempt {attempt}/{config.max_retries}: API error — {exc}")
            if attempt < config.max_retries:
                time.sleep(config.retry_delay * attempt)

    return None


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_dataset(config: GeneratorConfig, dummy: bool = False) -> None:
    """
    Generate n_samples traces from SEED_PROBLEMS and write to config.output_path.

    Cycles through seed problems if n_samples > len(SEED_PROBLEMS).
    If dummy=True, uses pre-built traces without any API call.
    """
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    seeds_cycle = SEED_PROBLEMS * ((config.n_samples // len(SEED_PROBLEMS)) + 1)
    seeds_cycle = seeds_cycle[: config.n_samples]

    generated = 0
    failed = 0

    with open(config.output_path, "w", encoding="utf-8") as fout:
        for i, seed in enumerate(seeds_cycle):
            print(f"[{i+1}/{config.n_samples}] domain={seed['domain']} ...", flush=True)

            if dummy:
                trace = _DUMMY_TRACES[i % len(_DUMMY_TRACES)]
            else:
                trace = generate_trace(seed["problem"], config)
                # Respect free-tier rate limit between requests (not on failures/retries)
                if trace is not None and config.request_delay > 0 and i < config.n_samples - 1:
                    time.sleep(config.request_delay)

            if trace is None:
                print(f"  FAILED — skipping")
                failed += 1
                continue

            record = {
                "problem": seed["problem"],
                "trace": trace,
                "source": "synthetic",
                "domain": seed["domain"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            generated += 1
            print(f"  OK ({generated} saved)")

    print(f"\nDone: {generated} generated, {failed} failed → {config.output_path}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _run_smoke_test() -> None:
    print("Running generate_synthetic smoke test...\n")

    # Check that API key env var is readable (may be unset — just warn)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not any([gemini_key, anthropic_key, openai_key]):
        print("  WARNING: No API keys found (GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY).")
        print("  Add keys to .env — see .env.example for the format.")
        print("  Free Gemini key: https://aistudio.google.com/app/apikey")
    else:
        providers_ready = []
        if gemini_key: providers_ready.append("Gemini ✓")
        if anthropic_key: providers_ready.append("Anthropic ✓")
        if openai_key: providers_ready.append("OpenAI ✓")
        print(f"  API keys found: {', '.join(providers_ready)}")

    # Validate all seed problems are present
    assert len(SEED_PROBLEMS) == 100, f"Expected 100 seed problems, got {len(SEED_PROBLEMS)}"
    domains = [p["domain"] for p in SEED_PROBLEMS]
    for d in ["math", "code_debug", "factual_qa", "data_analysis"]:
        count = domains.count(d)
        assert count == 25, f"Expected 25 {d} problems, got {count}"
    print(f"  Seed problems: {len(SEED_PROBLEMS)} (25 per domain) ✓")

    # Validate the trace validator itself
    valid_trace = (
        '<tool_call>{"name":"python_repl","arguments":{"code":"print(2+2)"}}</tool_call>\n'
        '<tool_response>{"result":"4","success":true}</tool_response>\n'
        "<final_answer>4</final_answer>"
    )
    ok, reason = validate_trace(valid_trace)
    assert ok, f"Valid trace was rejected: {reason}"
    print(f"  Trace validator (valid case): PASS ✓")

    invalid_trace = "The answer is 4 because 2+2=4."
    ok2, _ = validate_trace(invalid_trace)
    assert not ok2
    print(f"  Trace validator (invalid case): PASS ✓")

    # Config default instantiation
    cfg = GeneratorConfig()
    assert cfg.n_samples == 100
    print(f"  GeneratorConfig default: PASS ✓")

    print("\ngenerate_synthetic smoke test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Manthan synthetic data generator")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests and exit (no API)")
    parser.add_argument("--dummy-data", action="store_true", help="Generate dummy traces without API (for testing pipeline)")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of traces to generate")
    parser.add_argument("--output", type=Path, default=Path("data/raw/synthetic_traces.jsonl"))
    parser.add_argument("--provider", choices=["gemini", "anthropic", "openai"], default="gemini")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model name for the chosen provider")
    parser.add_argument("--request-delay", type=float, default=13.0,
                        help="Seconds between API requests (default 13s for gemini-2.5-flash free tier 5 RPM)")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        sys.exit(0)

    cfg = GeneratorConfig(
        output_path=args.output,
        n_samples=args.n_samples,
        provider=args.provider,
        model=args.model,
        request_delay=args.request_delay,
    )
    generate_dataset(cfg, dummy=args.dummy_data)
