# techniques/self_consistency.py
from collections import Counter
from techniques.cot import run_cot

def run_self_consistency(question: str, samples: int = 5) -> str:
    answers = [run_cot(question) for _ in range(samples)]
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common