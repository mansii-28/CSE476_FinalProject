"""
agent.py — Core agent that routes questions to the right inference-time technique.

Each domain maps to one or more techniques. The agent's solve() method is the
single entry point called by main.py for every instance.
"""

from techniques.cot             import run_cot
from techniques.self_consistency import run_self_consistency
from techniques.tot             import run_tot
from techniques.self_refine     import run_self_refine
from techniques.react           import run_react
from techniques.decomposition   import run_decomposition
from techniques.least_to_most   import run_least_to_most
from techniques.tool_reasoning  import run_tool_reasoning


class Agent:
    """
    Orchestrates inference-time techniques based on question domain.

    Usage:
        agent = Agent()
        prediction, technique_name = agent.solve(question, domain="math")
    """

    def solve(self, question: str, domain: str = "unknown") -> tuple[str, str]:
        """
        Route a question to the appropriate technique and return:
            (prediction, technique_name)

        technique_name is logged by main.py for the domain summary report.
        """
        # Simple routing logic based on domain
        if domain == "math":
            return self._tool_reasoning(question)
        elif domain == "common_sense":
            # Maps to least_to_most based on original comment
            return self._least_to_most(question)
        elif domain == "future_prediction":
            # Maps to react based on original comment
            return self._react(question)
        elif domain == "coding":
            return self._self_refine(question)
        elif domain == "planning":
            return self._tree_of_thought(question)
            

        # Fallback to chain-of-thought
        return self._chain_of_thought(question)

    # ---------------------------------------------------------------------------
    # One private method per technique — each wraps the imported function and
    # returns (prediction, technique_name) so solve() always has a uniform return.
    # ---------------------------------------------------------------------------

    def _chain_of_thought(self, question: str) -> tuple[str, str]:
        """
        Prompt the model to reason step by step before giving a final answer.
        Best for: math, common_sense, coding.
        """
        prediction = run_cot(question)
        return prediction, "cot"

    def _self_consistency(self, question: str) -> tuple[str, str]:
        """
        Sample the model multiple times and take the majority-vote answer.
        Best for: math (where answers converge), common_sense.
        """
        prediction = run_self_consistency(question)
        return prediction, "self_consistency"

    def _tree_of_thought(self, question: str) -> tuple[str, str]:
        """
        Explore multiple reasoning branches, evaluate each, and prune dead ends.
        Best for: planning, multi-step logic puzzles.
        """
        prediction = run_tot(question)
        return prediction, "tot"

    def _self_refine(self, question: str) -> tuple[str, str]:
        """
        Generate an answer, critique it, then revise. Loop until stable.
        Best for: coding, open-ended tasks where quality can be judged.
        """
        prediction = run_self_refine(question)
        return prediction, "self_refine"

    def _react(self, question: str) -> tuple[str, str]:
        """
        Interleave reasoning steps with tool calls (e.g. search), observe
        results, and continue reasoning. Best for: future_prediction.
        """
        prediction = run_react(question)
        return prediction, "react"

    def _tool_reasoning(self, question: str) -> tuple[str, str]:
        """
        Simpler formulation that can use tools. 
        Best for: questions mostly solvable but maybe needing a quick calculation or fact check.
        """
        prediction = run_tool_reasoning(question)
        return prediction, "tool_reasoning"

    def _decomposition(self, question: str) -> tuple[str, str]:
        """
        Break the question into sub-problems, solve each, then recombine.
        Best for: planning, multi-hop math.
        """
        prediction = run_decomposition(question)
        return prediction, "decomposition"

    def _least_to_most(self, question: str) -> tuple[str, str]:
        """
        Ask the model what simpler questions need to be solved first,
        answer them in order, and use each answer to solve the next.
        Best for: common_sense, chained reasoning.
        """
        prediction = run_least_to_most(question)
        return prediction, "least_to_most"