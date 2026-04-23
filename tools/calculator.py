def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""

    characters = "1234567890+-*.()%/={}[]<>^! "
    for character in expression:
        if character not in characters:
            return "Error: Invalid character in expression"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
