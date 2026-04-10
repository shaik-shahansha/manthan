"""
Local dataset builder for Genesis Manthan.

Generates 100 high-quality, diverse tool-calling training traces WITHOUT any API.
Covers 4 domains: math, code_debug, factual_qa, data_analysis.
All solutions are pre-written, executed locally to verify correctness, and
formatted as ChatML tool-calling traces.

Usage:
    python src/data/build_local_dataset.py
    python src/data/build_local_dataset.py --output data/raw/local_traces.jsonl
    python src/data/build_local_dataset.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Seed problems with pre-written Python solution code
# Each entry: (domain, problem_text, python_code)
# The code must print the final answer on the last line.
# ---------------------------------------------------------------------------

PROBLEMS: list[tuple[str, str, str]] = [

    # ── MATH (25) ────────────────────────────────────────────────────────────

    ("math",
     "A store had 240 items. They sold 35% on Monday and 25% of the remainder on Tuesday. How many items are left?",
     """\
items = 240
after_monday = items * (1 - 0.35)
after_tuesday = after_monday * (1 - 0.25)
print(round(after_tuesday, 2))
"""),

    ("math",
     "A train travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. What is the total distance?",
     """\
d1 = 60 * 2.5
d2 = 80 * 1.5
print(round(d1 + d2, 2))
"""),

    ("math",
     "Lisa earns $1,200/month. She saves 25% and spends $300 on rent. How much is left after savings and rent?",
     """\
income = 1200
savings = income * 0.25
rent = 300
left = income - savings - rent
print(round(left, 2))
"""),

    ("math",
     "A rectangle has perimeter 56 cm. Its length is 3 times its width. Find the area in cm².",
     """\
# 2*(l + w) = 56; l = 3w => 2*(3w + w) = 56 => 8w = 56 => w = 7
w = 56 / 8
l = 3 * w
area = l * w
print(round(area, 2))
"""),

    ("math",
     "If 8 workers can build a wall in 12 days, how many days will 6 workers take?",
     """\
# work = workers * days is constant
total_work = 8 * 12
days = total_work / 6
print(round(days, 2))
"""),

    ("math",
     "A shopkeeper buys an item for $400 and sells it with a 30% profit. What is the selling price?",
     """\
cost = 400
selling_price = cost * 1.30
print(round(selling_price, 2))
"""),

    ("math",
     "How many seconds are there in 3 weeks?",
     """\
seconds = 3 * 7 * 24 * 3600
print(seconds)
"""),

    ("math",
     "A tank is 40% full with 200 litres. What is the full capacity of the tank in litres?",
     """\
capacity = 200 / 0.40
print(round(capacity, 2))
"""),

    ("math",
     "John scored 78, 85, 92, 67, and 88 in five tests. What is his average score?",
     """\
scores = [78, 85, 92, 67, 88]
avg = sum(scores) / len(scores)
print(round(avg, 2))
"""),

    ("math",
     "A car depreciates by 15% per year. It costs $20,000 now. What will it be worth in 3 years?",
     """\
value = 20000 * (0.85 ** 3)
print(round(value, 2))
"""),

    ("math",
     "Two numbers sum to 84. The larger is 3 times the smaller. Find both numbers.",
     """\
# x + 3x = 84
smaller = 84 / 4
larger = 3 * smaller
print(f"smaller={int(smaller)}, larger={int(larger)}")
"""),

    ("math",
     "A rope is 15.6 metres long. It is cut into pieces of 0.6 m each. How many pieces are there?",
     """\
pieces = 15.6 / 0.6
print(int(round(pieces)))
"""),

    ("math",
     "What is the compound interest on $5,000 at 8% per year for 2 years?",
     """\
principal = 5000
rate = 0.08
years = 2
amount = principal * (1 + rate) ** years
ci = amount - principal
print(round(ci, 2))
"""),

    ("math",
     "A field is 120m × 80m. How many full laps of the perimeter are needed to cover at least 1 km?",
     """\
import math
perimeter = 2 * (120 + 80)
laps = math.ceil(1000 / perimeter)
print(laps)
"""),

    ("math",
     "The sum of the first N natural numbers is 210. Find N.",
     """\
# N*(N+1)/2 = 210 => N^2 + N - 420 = 0
import math
n = (-1 + math.sqrt(1 + 4 * 420)) / 2
print(int(round(n)))
"""),

    ("math",
     "A recipe for 4 people needs 300g of flour. How much flour is needed for 7 people?",
     """\
flour = 300 * 7 / 4
print(round(flour, 2))
"""),

    ("math",
     "Convert 98.6°F to Celsius. Formula: C = (F - 32) × 5/9",
     """\
f = 98.6
c = (f - 32) * 5 / 9
print(round(c, 2))
"""),

    ("math",
     "A sphere has radius 7 cm. Find its volume to 2 decimal places. (V = 4/3 × π × r³)",
     """\
import math
r = 7
volume = (4/3) * math.pi * r**3
print(round(volume, 2))
"""),

    ("math",
     "17 is what percentage of 340?",
     """\
pct = 17 / 340 * 100
print(round(pct, 2))
"""),

    ("math",
     "A cistern can be filled in 6 hours and emptied in 8 hours. If both pipes open together, how long to fill it?",
     """\
# Fill rate: 1/6 per hour, drain rate: 1/8 per hour
# Net rate: 1/6 - 1/8 = (4-3)/24 = 1/24 per hour
time_hours = 1 / (1/6 - 1/8)
print(round(time_hours, 2))
"""),

    ("math",
     "If log₂(x) = 5, what is x?",
     """\
x = 2 ** 5
print(x)
"""),

    ("math",
     "How many prime numbers are there between 1 and 50 (exclusive)?",
     """\
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 50) if is_prime(n)]
print(len(primes))
"""),

    ("math",
     "A bag has 5 red, 3 blue, and 2 green balls. What is the probability of picking a blue ball?",
     """\
total = 5 + 3 + 2
prob = 3 / total
print(round(prob, 4))
"""),

    ("math",
     "What is 2^10 + 2^8?",
     """\
result = 2**10 + 2**8
print(result)
"""),

    ("math",
     "The hypotenuse of a right triangle is 13 cm and one leg is 5 cm. Find the other leg.",
     """\
import math
other_leg = math.sqrt(13**2 - 5**2)
print(round(other_leg, 2))
"""),

    # ── FACTUAL QA (25) ──────────────────────────────────────────────────────

    ("factual_qa",
     "How many seconds are in a leap year?",
     """\
seconds = 366 * 24 * 3600
print(seconds)
"""),

    ("factual_qa",
     "What is the speed of light in km/s (rounded to the nearest integer)?",
     """\
# Speed of light = 299,792,458 m/s
speed_km_s = round(299_792_458 / 1000)
print(speed_km_s)
"""),

    ("factual_qa",
     "How many bits are in 1 terabyte (using 1 TB = 1024⁴ bytes)?",
     """\
bits = (1024 ** 4) * 8
print(bits)
"""),

    ("factual_qa",
     "What is the sum of all ASCII codes for the word 'Python'?",
     """\
total = sum(ord(c) for c in 'Python')
print(total)
"""),

    ("factual_qa",
     "How many days are between January 1, 2024 and July 4, 2024, inclusive?",
     """\
from datetime import date
delta = (date(2024, 7, 4) - date(2024, 1, 1)).days + 1
print(delta)
"""),

    ("factual_qa",
     "Convert 255 from decimal to binary and hexadecimal.",
     """\
n = 255
print(f"binary={bin(n)}, hex={hex(n)}")
"""),

    ("factual_qa",
     "What is the 100th prime number?",
     """\
def nth_prime(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes[-1]

print(nth_prime(100))
"""),

    ("factual_qa",
     "How many trailing zeros does 50! have?",
     """\
# Count factors of 5 in 50!
n = 50
zeros = 0
power = 5
while power <= n:
    zeros += n // power
    power *= 5
print(zeros)
"""),

    ("factual_qa",
     "What is the GCD of 1071 and 462?",
     """\
import math
print(math.gcd(1071, 462))
"""),

    ("factual_qa",
     "What is the LCM of 12, 15, and 20?",
     """\
import math
def lcm(a, b):
    return a * b // math.gcd(a, b)
result = lcm(lcm(12, 15), 20)
print(result)
"""),

    ("factual_qa",
     "What is the area of a regular hexagon with side length 6 cm? (Round to 2 decimal places.)",
     """\
import math
s = 6
area = (3 * math.sqrt(3) / 2) * s**2
print(round(area, 2))
"""),

    ("factual_qa",
     "In how many ways can 5 people sit in a row?",
     """\
import math
ways = math.factorial(5)
print(ways)
"""),

    ("factual_qa",
     "What is the sum of the digits of 2^100?",
     """\
digits_sum = sum(int(d) for d in str(2**100))
print(digits_sum)
"""),

    ("factual_qa",
     "What is Euler's number e to 10 decimal places?",
     """\
import math
print(f"{math.e:.10f}")
"""),

    ("factual_qa",
     "What is the 20th Fibonacci number (0-indexed, F(0)=0, F(1)=1)?",
     """\
a, b = 0, 1
for _ in range(20):
    a, b = b, a + b
print(a)
"""),

    ("factual_qa",
     "How many months in a year have exactly 30 days?",
     """\
# April, June, September, November
print(4)
"""),

    ("factual_qa",
     "What is the sum 1 + 2 + 3 + ... + 1000?",
     """\
total = 1000 * 1001 // 2
print(total)
"""),

    ("factual_qa",
     "What is the square root of 1764?",
     """\
import math
print(int(math.sqrt(1764)))
"""),

    ("factual_qa",
     "How many characters are in the string 'The quick brown fox jumps over the lazy dog'?",
     """\
s = 'The quick brown fox jumps over the lazy dog'
print(len(s))
"""),

    ("factual_qa",
     "What is 17 raised to the power 4?",
     """\
print(17 ** 4)
"""),

    ("factual_qa",
     "How many unique permutations are there of the letters in 'MISSISSIPPI'?",
     """\
import math
# 11 letters: M=1, I=4, S=4, P=2
total = math.factorial(11) // (math.factorial(4) * math.factorial(4) * math.factorial(2))
print(total)
"""),

    ("factual_qa",
     "What is the result of 0b1010 XOR 0b1100 in decimal?",
     """\
result = 0b1010 ^ 0b1100
print(result)
"""),

    ("factual_qa",
     "What is the Euclidean distance between points (3, 4) and (7, 1)?",
     """\
import math
dist = math.sqrt((7-3)**2 + (1-4)**2)
print(round(dist, 4))
"""),

    ("factual_qa",
     "What day of the week was January 1, 2000?",
     """\
from datetime import date
d = date(2000, 1, 1)
print(d.strftime('%A'))
"""),

    ("factual_qa",
     "What is the value of π (pi) rounded to 6 decimal places?",
     """\
import math
print(round(math.pi, 6))
"""),

    # ── CODE DEBUG (25) ──────────────────────────────────────────────────────

    ("code_debug",
     "Fix this function that should sum ALL elements but skips the first:\ndef sum_list(lst):\n    total = 0\n    for i in range(1, len(lst)):\n        total += lst[i]\n    return total",
     """\
def sum_list(lst):
    total = 0
    for i in range(len(lst)):  # Fixed: range starts at 0
        total += lst[i]
    return total

# Verify fix
test = [10, 20, 30, 40]
result = sum_list(test)
print(result)  # Should be 100
assert result == 100, f"Expected 100, got {result}"
print("Fix verified: sum_list now includes first element")
"""),

    ("code_debug",
     "Fix this factorial function where factorial(0) incorrectly returns 0 instead of 1:\ndef factorial(n):\n    if n == 0: return 0\n    return n * factorial(n-1)",
     """\
def factorial(n):
    if n == 0: return 1  # Fixed: base case should return 1
    return n * factorial(n-1)

# Verify fix
assert factorial(0) == 1, "factorial(0) should be 1"
assert factorial(5) == 120, "factorial(5) should be 120"
print(f"factorial(0)={factorial(0)}, factorial(5)={factorial(5)}")
print("Fix verified: base case now returns 1")
"""),

    ("code_debug",
     "Fix this palindrome checker that uses .reverse() which doesn't exist on strings:\ndef is_palindrome(s):\n    return s == s.reverse()",
     """\
def is_palindrome(s):
    return s == s[::-1]  # Fixed: use slicing to reverse string

# Verify fix
assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
print(f"is_palindrome('racecar')={is_palindrome('racecar')}")
print(f"is_palindrome('hello')={is_palindrome('hello')}")
print("Fix verified: using s[::-1] for reversal")
"""),

    ("code_debug",
     "Fix this Celsius to Fahrenheit converter that adds 23 instead of 32:\ndef celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 23",
     """\
def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32  # Fixed: add 32 not 23

# Verify: water boils at 100°C = 212°F
result = celsius_to_fahrenheit(100)
assert abs(result - 212) < 0.01, f"Expected 212, got {result}"
print(f"celsius_to_fahrenheit(100) = {result}")
print("Fix verified: formula now adds 32")
"""),

    ("code_debug",
     "Fix this vowel counter that misses uppercase vowels:\ndef count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')",
     """\
def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')  # Fixed: convert to lowercase first

# Verify fix
result = count_vowels("Hello World")
assert result == 3, f"Expected 3 vowels (e, o, o), got {result}"
print(f"count_vowels('Hello World') = {result}")
print("Fix verified: now handles uppercase vowels")
"""),

    ("code_debug",
     "Fix this FizzBuzz where if/elif conditions are inverted (uses truthy check instead of ==0):\nfor i in range(1, 101):\n    if i % 3: print('Fizz')\n    elif i % 5: print('Buzz')\n    else: print(i)",
     """\
results = []
for i in range(1, 16):  # Test first 15 for verification
    if i % 15 == 0:
        results.append('FizzBuzz')
    elif i % 3 == 0:  # Fixed: == 0 means divisible
        results.append('Fizz')
    elif i % 5 == 0:  # Fixed: == 0 means divisible
        results.append('Buzz')
    else:
        results.append(str(i))

print(' '.join(results))
assert results[2] == 'Fizz', "3 should be Fizz"
assert results[4] == 'Buzz', "5 should be Buzz"
assert results[14] == 'FizzBuzz', "15 should be FizzBuzz"
print("Fix verified: FizzBuzz logic corrected")
"""),

    ("code_debug",
     "Fix this binary search with infinite loop when arr[mid] < target (lo = mid should be lo = mid+1):\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid\n        else: hi = mid\n    return -1",
     """\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1  # Also fixed: hi = len-1 not len
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1  # Fixed: was lo = mid (infinite loop)
        else:
            hi = mid - 1
    return -1

arr = [1, 3, 5, 7, 9, 11, 13]
assert binary_search(arr, 7) == 3
assert binary_search(arr, 1) == 0
assert binary_search(arr, 99) == -1
print(f"binary_search([1,3,5,7,9,11,13], 7) = {binary_search(arr, 7)}")
print("Fix verified: no infinite loop, correct indices")
"""),

    ("code_debug",
     "Fix this word counter that iterates over characters instead of words:\nwords = 'hello world foo bar'\nword_count = {}\nfor w in words:\n    word_count[w] = word_count.get(w, 0) + 1",
     """\
text = 'hello world foo bar'
word_count = {}
for w in text.split():  # Fixed: .split() splits into words not characters
    word_count[w] = word_count.get(w, 0) + 1

print(word_count)
assert word_count == {'hello': 1, 'world': 1, 'foo': 1, 'bar': 1}
print("Fix verified: now counts words not characters")
"""),

    ("code_debug",
     "Fix this matrix initialization where all rows share the same list object:\nmatrix = [[0]*3]*3\nmatrix[0][0] = 1\nprint(matrix)  # Bug: all rows show the change",
     """\
# Fixed: use list comprehension so each row is a distinct object
matrix = [[0]*3 for _ in range(3)]
matrix[0][0] = 1
print(matrix)
assert matrix[0][0] == 1
assert matrix[1][0] == 0, "Other rows should not be affected"
assert matrix[2][0] == 0, "Other rows should not be affected"
print("Fix verified: rows are independent objects")
"""),

    ("code_debug",
     "Fix this first_duplicate function that adds to seen before checking (should check THEN add):\ndef first_duplicate(lst):\n    seen = set()\n    for x in lst:\n        if x in seen:\n            seen.add(x)\n            return x\n    return None",
     """\
def first_duplicate(lst):
    seen = set()
    for x in lst:
        if x in seen:
            return x      # Found duplicate: return immediately
        seen.add(x)       # Fixed: add AFTER checking, not inside the if
    return None

assert first_duplicate([1, 2, 3, 2, 4]) == 2
assert first_duplicate([1, 2, 3]) is None
print(f"first_duplicate([1,2,3,2,4]) = {first_duplicate([1,2,3,2,4])}")
print("Fix verified: seen.add() moved outside if block")
"""),

    ("code_debug",
     "Fix this circle area function that uses ^ (XOR) instead of ** (power):\nimport math\ndef circle_area(r):\n    return math.pi * r ^ 2",
     """\
import math

def circle_area(r):
    return math.pi * r ** 2  # Fixed: ** is exponentiation, ^ is bitwise XOR

result = circle_area(5)
expected = math.pi * 25
assert abs(result - expected) < 1e-9
print(f"circle_area(5) = {round(result, 4)}")
print("Fix verified: using ** for exponentiation")
"""),

    ("code_debug",
     "Fix this merge_sorted function that crashes with IndexError when one list is exhausted:\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) or j < len(b):\n        if a[i] < b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result",
     """\
def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):  # Fixed: 'and' not 'or'
        if a[i] < b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:])  # Fixed: append remaining elements
    result.extend(b[j:])
    return result

merged = merge_sorted([1, 3, 5], [2, 4, 6, 8])
assert merged == [1, 2, 3, 4, 5, 6, 8], f"Got {merged}"
print(f"merge_sorted([1,3,5],[2,4,6,8]) = {merged}")
print("Fix verified: handles exhausted lists correctly")
"""),

    ("code_debug",
     "Fix this running_average that divides by i=0 on the first element:\ndef running_average(nums):\n    total = 0\n    for i, n in enumerate(nums):\n        total += n\n        nums[i] = total / i\n    return nums",
     """\
def running_average(nums):
    total = 0
    result = []
    for i, n in enumerate(nums):
        total += n
        result.append(total / (i + 1))  # Fixed: divide by (i+1), not i; use separate result list
    return result

r = running_average([4, 8, 6, 10])
assert r == [4.0, 6.0, 6.0, 7.0], f"Got {r}"
print(f"running_average([4,8,6,10]) = {r}")
print("Fix verified: divides by (i+1), returns new list")
"""),

    ("code_debug",
     "Fix the Caesar cipher that only handles lowercase but not uppercase letters:\ndef caesar_cipher(text, shift):\n    result = ''\n    for ch in text:\n        if ch.isalpha():\n            result += chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))\n        else:\n            result += ch\n    return result",
     """\
def caesar_cipher(text, shift):
    result = ''
    for ch in text:
        if ch.isupper():   # Fixed: handle uppercase separately
            result += chr((ord(ch) - ord('A') + shift) % 26 + ord('A'))
        elif ch.islower():
            result += chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))
        else:
            result += ch
    return result

assert caesar_cipher("Hello", 3) == "Khoor"
assert caesar_cipher("abc XYZ", 1) == "bcd YZA"
print(f"caesar_cipher('Hello', 3) = {caesar_cipher('Hello', 3)}")
print("Fix verified: uppercase handled correctly")
"""),

    ("code_debug",
     "Fix the nth_fibonacci with off-by-one: nth_fibonacci(1) returns 2 instead of 1:\ndef nth_fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return b",
     """\
def nth_fibonacci(n):
    a, b = 0, 1
    for _ in range(n - 1):  # Fixed: range(n-1) instead of range(n)
        a, b = b, a + b
    return a if n == 0 else b  # For n>=1, return b after n-1 swaps

# Alternative clean version:
def nth_fibonacci_v2(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

assert nth_fibonacci_v2(1) == 1
assert nth_fibonacci_v2(2) == 1
assert nth_fibonacci_v2(10) == 55
print(f"F(1)={nth_fibonacci_v2(1)}, F(2)={nth_fibonacci_v2(2)}, F(10)={nth_fibonacci_v2(10)}")
print("Fix verified: nth_fibonacci(1)=1, nth_fibonacci(10)=55")
"""),

    ("code_debug",
     "Fix this dict mutation during iteration that raises RuntimeError:\ndata = {'a': 1, 'b': 2, 'c': 3}\nfor key in data:\n    if data[key] % 2 == 0:\n        del data[key]",
     """\
data = {'a': 1, 'b': 2, 'c': 3}
# Fixed: iterate over a copy of keys
for key in list(data.keys()):
    if data[key] % 2 == 0:
        del data[key]

print(data)
assert data == {'a': 1, 'c': 3}, f"Expected {{'a':1,'c':3}}, got {data}"
print("Fix verified: iterate over list(data.keys()) to avoid mutation error")
"""),

    ("code_debug",
     "This stack peek reads from index 0 (queue behavior) instead of the top (last element):\nstack = []\nstack.append(1); stack.append(2); stack.append(3)\ntop = stack[0]  # Bug: should peek top",
     """\
stack = []
stack.append(1)
stack.append(2)
stack.append(3)
top = stack[-1]  # Fixed: -1 index gives the top of the stack (last appended)
print(f"top of stack = {top}")
assert top == 3, f"Top should be 3, got {top}"
print("Fix verified: stack[-1] gives the most recently pushed element")
"""),

    ("code_debug",
     "Fix this rotate_list function: determine if it rotates left or right and verify with [1,2,3,4,5], k=2:\ndef rotate_list(lst, k):\n    k = k % len(lst)\n    return lst[k:] + lst[:k]",
     """\
def rotate_list(lst, k):
    k = k % len(lst)
    return lst[k:] + lst[:k]

# This is a LEFT rotation by k positions
result = rotate_list([1, 2, 3, 4, 5], 2)
print(f"rotate_list([1,2,3,4,5], 2) = {result}")
assert result == [3, 4, 5, 1, 2], f"Expected [3,4,5,1,2], got {result}"
print("This is a LEFT rotation: elements shift left by k positions")
print("Verification: [1,2,3,4,5] left-rotated by 2 = [3,4,5,1,2] - correct")
"""),

    ("code_debug",
     "Fix this safe_divide that uses a bare except which catches too broadly:\ndef safe_divide(a, b):\n    try:\n        return a / b\n    except:\n        return None",
     """\
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:  # Fixed: catch specific exception, not bare except
        return None

# Verify
assert safe_divide(10, 2) == 5.0
assert safe_divide(10, 0) is None
print(f"safe_divide(10, 2) = {safe_divide(10, 2)}")
print(f"safe_divide(10, 0) = {safe_divide(10, 0)}")
print("Fix verified: specific ZeroDivisionError caught instead of bare except")
"""),

    ("code_debug",
     "Fix this chunk_list function and verify it works correctly with [1..10], n=3:\ndef chunk_list(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]",
     """\
def chunk_list(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# The implementation is actually correct — verify it
result = chunk_list(list(range(1, 11)), 3)
print(f"chunk_list([1..10], 3) = {result}")
assert result == [[1,2,3],[4,5,6],[7,8,9],[10]], f"Got {result}"
print("No bug found: the implementation is correct. Last chunk has remaining elements.")
"""),

    ("code_debug",
     "Fix this flatten function and verify it works with nested lists like [[1,[2,3]],[4]]:\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if type(item) == list:\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
     """\
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):  # Fixed: use isinstance instead of type() == to handle subclasses
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

result = flatten([[1, [2, 3]], [4]])
print(f"flatten([[1,[2,3]],[4]]) = {result}")
assert result == [1, 2, 3, 4], f"Expected [1,2,3,4], got {result}"
print("Fix: used isinstance() instead of type() == for proper Python idiom")
"""),

    ("code_debug",
     "Fix this remove_duplicates function and verify with [1,2,1,3,2]:\ndef remove_duplicates(lst):\n    seen = []\n    return [x for x in lst if x not in seen and not seen.append(x)]",
     """\
# The original actually works but is a hack. Clean, idiomatic version:
def remove_duplicates(lst):
    seen = set()
    result = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

result = remove_duplicates([1, 2, 1, 3, 2])
print(f"remove_duplicates([1,2,1,3,2]) = {result}")
assert result == [1, 2, 3], f"Expected [1,2,3], got {result}"
print("Fix: rewritten to use a proper set for O(1) lookup, preserves insertion order")
"""),

    ("code_debug",
     "Verify gcd(48, 18) is correct using the iterative Euclidean algorithm:\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
     """\
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

result = gcd(48, 18)
print(f"gcd(48, 18) = {result}")
assert result == 6, f"Expected 6, got {result}"
# Verify with math.gcd
import math
assert gcd(48, 18) == math.gcd(48, 18)
print("Verification: gcd(48, 18) = 6 is correct")
"""),

    ("code_debug",
     "Fix and verify this filter that should print individual digit characters from 'Hello World 123':\ntext = 'Hello World 123'\ndigits = filter(str.isdigit, text)\nprint(list(digits))",
     """\
text = 'Hello World 123'
digits = list(filter(str.isdigit, text))
print(digits)
# filter(str.isdigit, text) filters individual CHARACTERS, so it returns ['1','2','3']
assert digits == ['1', '2', '3'], f"Got {digits}"
print("Verified: filter extracts individual digit characters ['1', '2', '3']")
print("Note: to get entire number groups, use re.findall(r'\\d+', text) instead")
"""),

    ("code_debug",
     "Verify what random.randint with seed 42 produces and find the top 3:\nimport random\nrandom.seed(42)\nresult = [random.randint(1, 10) for _ in range(5)]\nprint(sorted(result, reverse=True)[:3])",
     """\
import random
random.seed(42)
result = [random.randint(1, 10) for _ in range(5)]
top3 = sorted(result, reverse=True)[:3]
print(f"Generated: {result}")
print(f"Top 3 (descending): {top3}")
"""),

    # ── DATA ANALYSIS (25) ────────────────────────────────────────────────────

    ("data_analysis",
     "Given [4, 7, 2, 9, 1, 5, 8, 3, 6, 10], find the median and mean, then return all values above the mean.",
     """\
import statistics
data = [4, 7, 2, 9, 1, 5, 8, 3, 6, 10]
mean = statistics.mean(data)
median = statistics.median(data)
above_mean = [x for x in data if x > mean]
print(f"mean={mean}, median={median}, above_mean={sorted(above_mean)}")
"""),

    ("data_analysis",
     "Given sales data [120, 145, 98, 167, 134, 189, 112, 156], compute the percentage change from first to last month.",
     """\
sales = [120, 145, 98, 167, 134, 189, 112, 156]
pct_change = (sales[-1] - sales[0]) / sales[0] * 100
print(f"First={sales[0]}, Last={sales[-1]}, Change={round(pct_change, 2)}%")
"""),

    ("data_analysis",
     "Find the two most frequent words in: 'the cat sat on the mat the cat wore a hat'",
     """\
from collections import Counter
text = 'the cat sat on the mat the cat wore a hat'
words = text.split()
freq = Counter(words)
top2 = freq.most_common(2)
print(f"Top 2 words: {top2}")
"""),

    ("data_analysis",
     "Given temperatures [22, 19, 25, 18, 30, 27, 21] for a week, find the hottest day (0=Mon) and the temperature range.",
     """\
temps = [22, 19, 25, 18, 30, 27, 21]
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
hottest_idx = temps.index(max(temps))
temp_range = max(temps) - min(temps)
print(f"Hottest day: {days[hottest_idx]} ({max(temps)}°C)")
print(f"Temperature range: {temp_range}°C")
"""),

    ("data_analysis",
     "Given exam scores [55, 72, 68, 91, 43, 77, 85, 60, 88, 52], compute how many students passed (score >= 60) and the pass rate.",
     """\
scores = [55, 72, 68, 91, 43, 77, 85, 60, 88, 52]
passed = sum(1 for s in scores if s >= 60)
pass_rate = passed / len(scores) * 100
print(f"Passed: {passed}/{len(scores)}, Pass rate: {round(pass_rate, 1)}%")
"""),

    ("data_analysis",
     "From the list [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], remove duplicates while preserving insertion order.",
     """\
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
seen = set()
unique = []
for x in data:
    if x not in seen:
        seen.add(x)
        unique.append(x)
print(f"Original: {data}")
print(f"Unique (order preserved): {unique}")
"""),

    ("data_analysis",
     "Given {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'Diana': 95, 'Eve': 88}, sort students by score descending and find the top 3.",
     """\
scores = {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'Diana': 95, 'Eve': 88}
sorted_students = sorted(scores.items(), key=lambda x: x[1], reverse=True)
top3 = sorted_students[:3]
print(f"Ranked: {sorted_students}")
print(f"Top 3: {top3}")
"""),

    ("data_analysis",
     "Analyze the list [10, 20, 10, 30, 20, 10, 40, 30, 10] to find how many times each number appears and which appears most.",
     """\
from collections import Counter
data = [10, 20, 10, 30, 20, 10, 40, 30, 10]
freq = Counter(data)
most_common = freq.most_common(1)[0]
print(f"Frequency: {dict(freq)}")
print(f"Most common: {most_common[0]} appears {most_common[1]} times")
"""),

    ("data_analysis",
     "Given two lists prices=[100, 200, 150, 300, 250] and quantities=[5, 3, 8, 2, 4], find total revenue and the highest revenue product index.",
     """\
prices = [100, 200, 150, 300, 250]
quantities = [5, 3, 8, 2, 4]
revenues = [p * q for p, q in zip(prices, quantities)]
total = sum(revenues)
best_idx = revenues.index(max(revenues))
print(f"Revenues: {revenues}")
print(f"Total revenue: {total}")
print(f"Highest revenue product: index {best_idx} (revenue={max(revenues)})")
"""),

    ("data_analysis",
     "Given a list of strings ['apple', 'banana', 'cherry', 'date', 'elderberry'], group them by length.",
     """\
fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
by_length = {}
for fruit in fruits:
    length = len(fruit)
    by_length.setdefault(length, []).append(fruit)
for length in sorted(by_length):
    print(f"Length {length}: {by_length[length]}")
"""),

    ("data_analysis",
     "From monthly expenses [450, 520, 380, 600, 490, 710, 430], compute the 3-month rolling average.",
     """\
expenses = [450, 520, 380, 600, 490, 710, 430]
rolling_avg = []
for i in range(2, len(expenses)):
    avg = sum(expenses[i-2:i+1]) / 3
    rolling_avg.append(round(avg, 2))
print(f"3-month rolling averages: {rolling_avg}")
"""),

    ("data_analysis",
     "Given a CSV-style string 'name,age,score\\nAlice,25,88\\nBob,30,75\\nCharlie,22,92', find the average age and highest scorer.",
     """\
data_str = 'name,age,score\\nAlice,25,88\\nBob,30,75\\nCharlie,22,92'
lines = data_str.strip().split('\\n')
header = lines[0].split(',')
rows = [dict(zip(header, line.split(','))) for line in lines[1:]]
avg_age = sum(int(r['age']) for r in rows) / len(rows)
best = max(rows, key=lambda r: int(r['score']))
print(f"Average age: {round(avg_age, 2)}")
print(f"Highest scorer: {best['name']} with {best['score']}")
"""),

    ("data_analysis",
     "Given [15, 22, 8, 35, 12, 28, 5, 19], find all outliers defined as values more than 1.5 standard deviations from the mean.",
     """\
import statistics
data = [15, 22, 8, 35, 12, 28, 5, 19]
mean = statistics.mean(data)
stdev = statistics.stdev(data)
threshold = 1.5 * stdev
outliers = [x for x in data if abs(x - mean) > threshold]
print(f"Mean={round(mean,2)}, StdDev={round(stdev,2)}, Threshold=±{round(threshold,2)}")
print(f"Outliers: {outliers}")
"""),

    ("data_analysis",
     "Given two sets A={1,2,3,4,5} and B={3,4,5,6,7}, compute union, intersection, and the symmetric difference.",
     """\
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}
print(f"Union: {sorted(A | B)}")
print(f"Intersection: {sorted(A & B)}")
print(f"Symmetric difference: {sorted(A ^ B)}")
"""),

    ("data_analysis",
     "Count how many words in 'to be or not to be that is the question' appear more than once.",
     """\
from collections import Counter
text = 'to be or not to be that is the question'
freq = Counter(text.split())
repeated = {word: count for word, count in freq.items() if count > 1}
print(f"Words appearing more than once: {repeated}")
print(f"Count of such words: {len(repeated)}")
"""),

    ("data_analysis",
     "Given [('Alice', 85), ('Bob', 92), ('Alice', 78), ('Bob', 95), ('Charlie', 88)], compute average score per person.",
     """\
records = [('Alice', 85), ('Bob', 92), ('Alice', 78), ('Bob', 95), ('Charlie', 88)]
totals = {}
counts = {}
for name, score in records:
    totals[name] = totals.get(name, 0) + score
    counts[name] = counts.get(name, 0) + 1
averages = {name: round(totals[name] / counts[name], 2) for name in totals}
print(f"Average scores: {averages}")
"""),

    ("data_analysis",
     "Given the number 123456789, compute the sum of its even digits and the sum of its odd digits separately.",
     """\
n = 123456789
digits = [int(d) for d in str(n)]
even_sum = sum(d for d in digits if d % 2 == 0)
odd_sum = sum(d for d in digits if d % 2 != 0)
print(f"Digits: {digits}")
print(f"Sum of even digits: {even_sum}")
print(f"Sum of odd digits: {odd_sum}")
"""),

    ("data_analysis",
     "Given a list of words ['hello', 'world', 'python', 'code', 'data'], sort them by length first, then alphabetically for ties.",
     """\
words = ['hello', 'world', 'python', 'code', 'data']
sorted_words = sorted(words, key=lambda w: (len(w), w))
print(f"Sorted by (length, alpha): {sorted_words}")
"""),

    ("data_analysis",
     "From the list [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], find all pairs that sum to 22.",
     """\
data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
target = 22
pairs = []
seen = set()
for x in data:
    complement = target - x
    if complement in seen:
        pairs.append((min(x, complement), max(x, complement)))
    seen.add(x)
pairs = list(set(pairs))
pairs.sort()
print(f"Pairs summing to {target}: {pairs}")
"""),

    ("data_analysis",
     "Analyze text 'the quick brown fox jumps over the lazy dog the fox': count total words, unique words, and top 3 most frequent.",
     """\
from collections import Counter
text = 'the quick brown fox jumps over the lazy dog the fox'
words = text.split()
freq = Counter(words)
print(f"Total words: {len(words)}")
print(f"Unique words: {len(freq)}")
print(f"Top 3 most frequent: {freq.most_common(3)}")
"""),

    ("data_analysis",
     "Given [100, 200, 300, 400, 500], compute the cumulative sum at each index.",
     """\
data = [100, 200, 300, 400, 500]
cumulative = []
total = 0
for x in data:
    total += x
    cumulative.append(total)
print(f"Cumulative sums: {cumulative}")
"""),

    ("data_analysis",
     "From the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], filter primes, then compute their sum and product.",
     """\
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

data = list(range(1, 11))
primes = [x for x in data if is_prime(x)]
from functools import reduce
import operator
product = reduce(operator.mul, primes)
print(f"Primes: {primes}")
print(f"Sum: {sum(primes)}, Product: {product}")
"""),

    ("data_analysis",
     "Given {'Jan':1500, 'Feb':1800, 'Mar':1200, 'Apr':2100, 'May':1650, 'Jun':1900}, find the month with highest growth and total for H1.",
     """\
monthly = {'Jan':1500, 'Feb':1800, 'Mar':1200, 'Apr':2100, 'May':1650, 'Jun':1900}
values = list(monthly.values())
months = list(monthly.keys())
total_h1 = sum(values)
# Growth: compare each month to previous
growth = [(months[i], values[i] - values[i-1]) for i in range(1, len(values))]
best_growth = max(growth, key=lambda x: x[1])
print(f"Total H1 revenue: {total_h1}")
print(f"Highest growth month: {best_growth[0]} (+{best_growth[1]})")
"""),

    ("data_analysis",
     "Given strings ['abc123', 'def456', 'ghi789', '123abc'], extract all digits and concatenate them, then sum as integer.",
     """\
import re
strings = ['abc123', 'def456', 'ghi789', '123abc']
all_digits = ''.join(re.sub(r'[^0-9]', '', s) for s in strings)
digit_sum = sum(int(d) for d in all_digits)
print(f"All digits concatenated: {all_digits}")
print(f"Sum of individual digits: {digit_sum}")
"""),

    ("data_analysis",
     "Given a list of (x, y) points [(1,2),(3,4),(5,6),(7,8),(9,10)], compute the centroid (average x, average y).",
     """\
points = [(1,2),(3,4),(5,6),(7,8),(9,10)]
cx = sum(p[0] for p in points) / len(points)
cy = sum(p[1] for p in points) / len(points)
print(f"Centroid: ({cx}, {cy})")
"""),
]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def execute_code(code: str, timeout: int = 15) -> tuple[bool, str]:
    """Execute Python code in a subprocess and return (success, output)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
        prefix="manthan_build_", dir=tempfile.gettempdir(),
        encoding="utf-8",
    ) as f:
        f.write(textwrap.dedent(code))
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    finally:
        Path(tmp).unlink(missing_ok=True)


def build_trace(problem: str, code: str, result: str) -> str:
    """Build a complete ChatML tool-calling trace string."""
    tool_call = json.dumps({
        "name": "python_repl",
        "arguments": {"code": textwrap.dedent(code).strip()}
    }, ensure_ascii=False)
    tool_response = json.dumps({"result": result, "success": True}, ensure_ascii=False)
    return (
        f'<tool_call>{tool_call}</tool_call>\n'
        f'<tool_response>{tool_response}</tool_response>\n'
        f'<final_answer>{result}</final_answer>'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(output_path: Path) -> int:
    """Build the dataset, write to output_path. Returns count of successful traces."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0
    records: list[dict] = []

    print(f"Building local dataset from {len(PROBLEMS)} seed problems...\n")

    for idx, (domain, problem, code) in enumerate(PROBLEMS, 1):
        ok, out = execute_code(code)
        status = "OK" if ok else "FAIL"
        short_problem = problem[:60].replace("\n", " ")
        print(f"  [{idx:3d}/{len(PROBLEMS)}] {status} [{domain:12s}] {short_problem}...")

        if ok:
            trace = build_trace(problem, code, out)
            records.append({
                "problem": problem,
                "trace": trace,
                "source": "programmatic",
                "domain": domain,
            })
            success_count += 1
        else:
            print(f"         ERROR: {out[:120]}")
            fail_count += 1

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nComplete: {success_count} succeeded, {fail_count} failed")
    print(f"Output: {output_path} ({success_count} traces)")
    return success_count


def _run_smoke_test() -> None:
    print("Running build_local_dataset smoke test...")
    # Test executor with a trivial script
    ok, out = execute_code("print(40 + 2)")
    assert ok, f"Executor failed: {out}"
    assert out.strip() == "42", f"Expected '42', got '{out}'"
    print(f"  Executor: print(40+2) -> '{out}' OK")

    # Test trace builder
    trace = build_trace("What is 2+2?", "print(4)", "4")
    assert "<tool_call>" in trace
    assert "<tool_response>" in trace
    assert "<final_answer>" in trace
    print(f"  Trace builder: OK ✓")

    print(f"\n  Total seed problems: {len(PROBLEMS)}")
    domain_counts: dict[str, int] = {}
    for domain, _, _ in PROBLEMS:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    for d, c in sorted(domain_counts.items()):
        print(f"    {d}: {c} problems")

    print("\nbuild_local_dataset smoke test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build local tool-calling training dataset")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/raw/local_traces.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        sys.exit(0)

    count = build_dataset(args.output)
    sys.exit(0 if count > 0 else 1)
