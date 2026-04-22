# test_cot.py
# usage: python -m tests/test_cot
from techniques.cot import run_cot

tests = [
    {
        "domain": "math",
        "input": "What is the sum of all integers from 1 to 100?",
        "expected": "5050",
    },
    {
        "domain": "common_sense",
        "input": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "expected": "Arthur's Magazine",
    },
    {
        "domain": "math",
        "input": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
        "expected": "150",
    },
]

for t in tests:
    result = run_cot(t["input"])
    passed = t["expected"].lower() in result.lower()
    mark = "✅" if passed else "❌"
    print(f"{mark} [{t['domain']}]")
    print(f"   expected : {t['expected']}")
    print(f"   got      : {result}")
    print()