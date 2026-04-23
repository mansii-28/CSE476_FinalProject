from api import call_model_messages
from tools.calculator import calculator
from tools.search import search
import json

TOOLS = {
    "calculator": calculator,
    "search": search
}

SYSTEM_PROMPT = """You are a smart and a supportive tool reasoning assistant.
You have access to the following tools:
1. calculator(expression): which evaluates mathematical expressions.
2. search(query): which looks up general knowledge facts.
Whenver you are using a tool or need to use a tool, you should output exactly the below given JSON format on a new line:
{"tool": "tool_name", "arguments": "tool_arguments"}
Wait for the tool result before providing your final answer. If you don't need the tools, just output your reasoning and the final answer directly.
If you have used tools, make sure your final response contains the final answer.
"""

def run_tool_reasoning(question: str, max_steps: int = 3) -> str:
    """
    Simple tool-using technique. 
    Allows the model to decide whether to call a tool or directly output an answer.
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
        trace += f"\nAssistant: {reply}"
        
        # Look for a tool call
        if '{"tool":' in reply:
            try:
                start = reply.find('{')
                end = reply.find('}') + 1

                str_json  = reply[start:end]
                data_tool = json.loads(str_json)

                tool_name = data_tool.get("tool")
                tool_args = data_tool.get("arguments")

                if tool_name in TOOLS:
                    result_tool = TOOLS[tool_name](tool_args)
                    messages.append({"role": "user", "content": f"Tool result: {result_tool}"})
                    trace += f"\nTool result: {result_tool}"
                else:
                    trace += f"\nError: Unknown tool '{tool_name}'"
                    break
            except Exception as e:
                trace += f"\nError parsing tool call: {e}"
                break
        else:
            trace += "\nFinal answer assumed since no tool call was detected."
            break
    
    return trace.strip()
                