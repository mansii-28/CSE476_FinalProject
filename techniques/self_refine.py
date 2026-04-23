# techniques/self_refine.py
from api import call_model
from techniques.cot import extract_final_answer

def run_self_refine(question: str, max_iterations: int = 2) -> str:
    # Step 1 — initial answer (same as CoT)
    result = call_model(
        prompt=question,
        system=(
            "You are a careful reasoning assistant. "
            "Think step by step, then give your final answer on a new line as:\n"
            "FINAL ANSWER: <answer>"
        ),
        temperature=0.0,
        max_tokens=1024,
    )
    answer = result["text"] if result["ok"] else ""

    # Step 2 — critique and refine
    for _ in range(max_iterations):
        critique = call_model(
            prompt=(
                f"Question: {question}\n\n"
                f"Proposed answer: {answer}\n\n"
                "Is this answer correct and complete? "
                "If yes, reply LGTM. "
                "If no, explain what's wrong and give a corrected answer ending with:\n"
                "FINAL ANSWER: <answer>"
            ),
            system="You are a strict critic. Be concise.",
            temperature=0.0,
            max_tokens=1024,
        )

        if not critique["ok"] or "LGTM" in critique["text"]:
            break

        answer = critique["text"]

    # Extract final answer
    return extract_final_answer(answer)