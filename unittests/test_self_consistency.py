#usage: pytest unittests/test_self_consistency.py -v
from techniques.self_consistency import run_self_consistency

def test_math_sum():
    result = run_self_consistency("What is the sum of all integers from 1 to 100?")
    assert "5050" in result

def test_math_multiply():
    result = run_self_consistency("What is 17 multiplied by 13?")
    assert "221" in result

def test_common_sense():
    result = run_self_consistency("Which magazine was started first, Arthur's Magazine or First for Women?")
    assert "arthur" in result.lower()