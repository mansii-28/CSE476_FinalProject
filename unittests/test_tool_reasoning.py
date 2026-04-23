# test tool reasoning technique
# usage: pytest unittests/test_tool_reasoning.py -v
from techniques.tool_reasoning import run_tool_reasoning

def test_calculator_tool():
    result = run_tool_reasoning("What is 20 * 5?")
    assert "100" in result

def test_search_tool():
    result = run_tool_reasoning("What is the capital of India?")
    assert "New Delhi" in result
