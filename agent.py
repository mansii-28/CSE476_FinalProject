"""
agent.py — Core agent that routes questions to the right inference-time technique.

Each domain maps to one or more techniques. The agent's solve() method is the
single entry point called by main.py and generate_answers.py for every instance.

Domain → technique mapping:
    math              → self_consistency
    common_sense      → least_to_most
    future_prediction → react
    coding            → self_refine
    planning          → tree_of_thought
    unknown           → detected via _detect_domain(), then routed above
"""

from techniques.cot              import run_cot
from techniques.self_consistency import run_self_consistency
from techniques.tot              import run_tot
from techniques.self_refine      import run_self_refine
from techniques.react            import run_react
from techniques.decomposition    import run_decomposition
from techniques.least_to_most    import run_least_to_most
from techniques.tool_reasoning   import run_tool_reasoning


class Agent:
    """
    Orchestrates inference-time techniques based on question domain.

    Usage:
        agent = Agent()
        prediction, technique_name = agent.solve(question, domain="math")
        prediction, technique_name = agent.solve(question)  # auto-detects domain
    """

    def solve(self, question: str, domain: str = "unknown") -> tuple[str, str]:
        """
        Route a question to the appropriate technique and return:
            (prediction, technique_name)

        If domain is "unknown" (e.g. test data has no domain field),
        _detect_domain() infers it from the question text.
        """
        if domain == "unknown" or not domain:
            domain = self._detect_domain(question)

        if domain == "math":
            return self._self_consistency(question)
        elif domain == "common_sense":
            return self._least_to_most(question)
        elif domain == "future_prediction":
            return self._react(question)
        elif domain == "coding":
            return self._self_refine(question)
        elif domain == "planning":
            return self._tree_of_thought(question)

        # Final fallback
        return self._chain_of_thought(question)

    # ---------------------------------------------------------------------------
    # Domain detector — used when test data has no "domain" field
    # ---------------------------------------------------------------------------

    def _detect_domain(self, question: str) -> str:
        """
        Heuristically infer domain from question text.
        Called automatically when domain="unknown".
        """
        q = question.lower()

        # Coding — function/code keywords
        if any(w in q for w in [
            "def ", "function", "implement", "algorithm", "class ",
            "return ", "code", "write a program", "write a function",
            "bug", "syntax", "compile", "runtime",
        ]):
            return "coding"

        # Planning — action/sequence/arrangement keywords
        if any(w in q for w in [
            "stack", "block", "move", "step", "plan", "sequence",
            "arrange", "order the", "sort the", "schedule",
            "first then", "before", "after",
        ]):
            return "planning"

        # Math — numeric calculation keywords
        if any(w in q for w in [
            "calculate", "compute", "solve", "equation", "percent", "%",
            "profit", "loss", "average", "sum of", "how many", "total",
            "how much", "price", "cost", "rate", "ratio", "probability",
            "multiply", "divide", "subtract", "add ", "integer", "prime",
        ]):
            return "math"

        # Future prediction — forecasting/estimation keywords
        if any(w in q for w in [
            "predict", "forecast", "will happen", "future", "estimate",
            "likely to", "next year", "by 2", "trend",
        ]):
            return "future_prediction"

        # Default — common sense covers most factual/trivia/multiple choice
        return "common_sense"

    # ---------------------------------------------------------------------------
    # One private method per technique
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