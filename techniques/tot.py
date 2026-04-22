from api import call_model
from techniques.cot import extract_final_answer

def run_tot(question: str) -> str:
    gen_result = call_model(
        prompt=question,
        system=(
            "You are an expert planner. Given a problem, propose 3 distinct, high-level"
            "step-by-step plans or approaches to solve it. Number them Plan 1, Plan 2 and Plan 3. "
            "Do not execute the plans just describe the approaches."
        ),
        temperature=0.7, 
        max_tokens=512,
    )
    if not gen_result["ok"]:
        return ""
        
    plans = gen_result["text"]
    eval_prompt = (
        f"Problem: {question}\n\n"
        f"Proposed Plans:\n{plans}\n\n"
        "Evaluate the pros and cons of each plan briefly."
        "then select the best plan and execute it step by step to solve the problem."
    )
    eval_result = call_model(
        prompt=eval_prompt,
        system=(
            "You are an expert evaluator and problem solver.Execute the best plan"
            "Then give your final answer on a new line as:\n"
            "FINAL ANSWER: <answer>"
        ),
        temperature=0.0,
        max_tokens=1024,
    )
    if not eval_result["ok"]:
        return ""

    return extract_final_answer(eval_result["text"])