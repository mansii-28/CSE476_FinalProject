# test react technique
# usage: pytest unittests/test_react.py -v
from techniques.react import run_react

def test_calculator_react():
    result = run_react("Calculate the result for 20 * 5.")
    assert "100" in result
    assert "Action:" in result

def test_search_react():
    result = run_react("What is the capital of India?")
    assert "New Delhi" in result
    assert "Action:" in result
