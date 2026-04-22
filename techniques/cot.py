# techniques/cot.py
from api import call_model

def run_cot(question: str) -> str:
    system = (
        "You are a careful reasoning assistant. "
        "Think through the problem step by step, then give your final answer "
        "on a new line in the format:\nFINAL ANSWER: <answer>"
    )

    result = call_model(
        prompt=question,
        system=system,
        temperature=0.0,
        max_tokens=1024,
    )

    if not result["ok"]:
        return ""

    return extract_final_answer(result["text"])


def extract_final_answer(text: str) -> str:
    """Pull out everything after 'FINAL ANSWER:' if present, else return full text."""
    marker = "FINAL ANSWER:"
    if marker in text:
        return text.split(marker)[-1].strip()
    return text.strip()