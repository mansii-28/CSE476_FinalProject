from api import call_model

DECOMPOSITION_SYSTEM = """You are a smart and a supportive assistant.
Your job is to break down the user's problem into a small set of subproblems which should be solved sequentially to get the final answer.
Also, do not solve the subproblems yet.
IMPORTANT NOTE FOR YOU: You must output ONLY the subproblems separated by a number following a dot then the number of the subproblem in words.
FOR EXAMPLE: 
1. First subproblem
2. Second subproblem
3. Third subproblem
and so on till the last subproblem.
"""

SOLVE_SYSTEM = """You are a smart and a supportive assistant.
You will be given the original problem, followed by a sequence of previously solved subproblems, and finally a current subproblem to solve.
Solve the current subproblem based on the previous settings. Please keep your explanation precise and to the point.
You should always end your response with exactly: "Answer: [your answer]"
"""

def run_least_to_most(question: str, max_subproblems: int = 5) -> str:
    """
    Least-to-Most reasoning framework.
    1. Decompose problem into subproblems.
    2. Solve subproblems sequentially using knowledge from earlier answers.
    Returns the final answer along with the decomposition and step-by-step trace.
    """
    # Decompose problem
    decomp_result = call_model(
        prompt=f"Original Problem: {question}\nDecompose this problem into simpler subproblems.",
        system=DECOMPOSITION_SYSTEM,
        temperature=0.0
    )
    
    if not decomp_result["ok"]:
        return f"Error during decomposition: {decomp_result['error']}"
        
    decomposition_text = decomp_result["text"]
    
    # Parse subproblems
    subproblems = []
    for new_line in decomposition_text.split('\n'):
        new_line = new_line.strip()
        if new_line and new_line[0].isdigit():
            subproblems.append(new_line)
            
    if not subproblems:
        return "Failed to decompose the problem into subproblems."
        
    subproblems = subproblems[:max_subproblems]
        
    trace = "Least-to-Most Trace:\n"
    trace += "Decomposition steps:\n"
    for i, sp in enumerate(subproblems, 1):
        trace += f"{i}. {sp}\n"
        
    # Solve sequentially
    context = ""
    for i, sp in enumerate(subproblems):
        prompt = f"Original Problem: {question}\n\n"
        if context:
            prompt += f"Previous context:\n{context}\n"
            
        prompt += f"Current Subproblem to solve: {sp}"
        
        solve_result = call_model(
            prompt=prompt,
            system=SOLVE_SYSTEM,
            temperature=0.0
        )
        
        if not solve_result["ok"]:
            trace += f"\nError solving subproblem '{sp}': {solve_result['error']}"
            break
            
        answer_text = solve_result["text"]
        trace += f"\nSubproblem {i+1}: {sp}\nResult: {answer_text}\n"
        
        # Accumulate context
        context += f"- Subproblem: {sp}\n- Answer: {answer_text}\n\n"
        
    trace += "\nFinal Process Completed."
    return trace.strip()
