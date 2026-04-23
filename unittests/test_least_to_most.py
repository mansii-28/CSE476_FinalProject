# test least to most technique
# usage: pytest unittests/test_least_to_most.py -v
from techniques.least_to_most import run_least_to_most

def test_l2m_simple_math():
    question = "I had 5 cats. 2 ran away, then I adopted 4 more. How many do I have left?"
    result = run_least_to_most(question)
    
    assert "Decomposition steps:" in result
    
    assert "7" in result
