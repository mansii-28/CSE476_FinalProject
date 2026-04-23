# test self refine technique
#usage: pytest unittests/test_self_refine.py -v
from techniques.self_refine import run_self_refine

def test_math_sum():
    result = run_self_refine("What is the sum of all integers from 1 to 100?")
    assert "5050" in result

def test_math_distance():
    result = run_self_refine("A train travels 60 miles per hour for 2.5 hours. How far does it travel?")
    assert "150" in result

def test_common_sense():
    result = run_self_refine("Which magazine was started first, Arthur's Magazine or First for Women?")
    assert "arthur" in result.lower()