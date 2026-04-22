# test chain of thought technique
# usage: pytest unittests/test_cot.py -v
from techniques.cot import run_cot

def test_math_sum():
    result = run_cot("What is the sum of all integers from 1 to 100?")
    assert "5050" in result

def test_math_distance():
    result = run_cot("A train travels 60 miles per hour for 2.5 hours. How far does it travel?")
    assert "150" in result

def test_common_sense():
    result = run_cot("Which magazine was started first, Arthur's Magazine or First for Women?")
    assert "arthur" in result.lower()