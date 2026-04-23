from api import call_model
from techniques.cot import extract_final_answer

def run_decomposition(question: str) -> str:
    decomp_result = call_model(
        prompt=question,
        system=(
            "You are an expert problem solver. Break down the following complex problem. "
            "Into 2 to 4 simpler sub questions that need to be answered to solve the main problem. "
            "List only the sub questions do not solve them yet."
        ),
        temperature=0.0,
        max_tokens=256,
    )
    
    if not decomp_result["ok"]:
        return ""  
    sub_questions = decomp_result["text"]
    solve_prompt = (
        f"Main Problem: {question}\n\n"
        f"Sub-questions to guide you:\n{sub_questions}\n\n"
        "Solve the main problem step-by-step by answering the sub-questions first."
    )
    solve_result = call_model(
        prompt=solve_prompt,
        system=(
            "You are a careful reasoning assistant. Think step by step. "
            "Then give your final answer on a new line as:\n"
            "FINAL ANSWER: <answer>"
        ),
        temperature=0.0,
        max_tokens=1024,
    )

    if not solve_result["ok"]:
        return ""
    return extract_final_answer(solve_result["text"])