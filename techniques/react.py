from api import call_model_messages
from tools.calculator import calculator
from tools.search import search
import json

TOOLS = {
    "calculator": calculator,
    "search": search
}

SYSTEM_PROMPT = """You are a very smart and a supportive assistant. 
You solve problems using the ReACT (Reasoning and Acting) framework.
You will have access to the following tools:
- calculator: which evaluates math expressions.
- search: which looks up facts.

You should use the following format to output your response for every step in the JSON format. Please do not wander around.

{
    "thought": "[Your reasoning for what you should do next]",
    "action": "[The name of the tool you are thinking to use: \"calculator\" or \"search\", or \"finish\" if done]",
    "action_input": "[The input for the tool, or the final answer if Action is finish]"
}

You will then receive an "Observation: [result]" from the environment. You can then continue with a new thought.
"""

def run_react(question: str, max_steps: int = 5) -> str:
    """
    Explicit Thought -> Action -> Observation loop.
    Returns the reasoning trace/history and the final answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    trace = ""
    for step in range(max_steps):
        result = call_model_messages(messages, temperature=0.0)
        if not result["ok"]:
            return f"{trace}\nError calling model: {result['error']}".strip()
        
        reply = result["text"]
        messages.append({"role": "assistant", "content": reply})
        trace += f"\n{reply}"
        
        action = None
        action_input = None
        
        lines = reply.split('\n')
        if '{' in reply:
            try:
                start = reply.find('{')
                end = reply.find('}') + 1

                str_json  = reply[start:end]
                data_tool = json.loads(str_json)

                tool_name = data_tool.get("action")
                tool_args = data_tool.get("action_input")

                if tool_name == "finish":
                    break

                if tool_name in TOOLS:
                    obs = TOOLS[tool_name](tool_args)
                else:
                    obs = f"Error: Tool '{tool_name}' not recognized."
                    
                obs_msg = f"Observation: {obs}"
                messages.append({"role": "user", "content": obs_msg})
                trace += f"\n{obs_msg}"
            except Exception as e:
                trace += f"\nError parsing tool call: {e}"
                break
        else:
            trace += "\nFinal answer assumed since no tool call was detected."
            break

    return trace.strip()
