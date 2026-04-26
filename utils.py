import re
import json

def normalize_answer(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, latex, and spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace('\\mathbf', '').replace('\\mathit', '').replace('$', '')
    text = re.sub(r'[^\w\s]', '', text)
    return "".join(text.split())

def extract_tool_json(text: str) -> dict:
    """Safely extracts a JSON object from text containing conversational context."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
