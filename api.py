"""
api.py — Thin wrapper around the ASU OpenAI-style /v1/chat/completions endpoint.

Every technique file imports call_model() from here — never copy-paste the
requests logic into individual technique files.

Environment variables (set in .env or shell):
    OPENAI_API_KEY   — Voyager Portal key
    API_BASE         — defaults to https://openai.rc.asu.edu/v1
    MODEL_NAME       — defaults to qwen3-30b-a3b-instruct-2507
"""

import os
import time
import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# ---------------------------------------------------------------------------
# Config — read once at import time
# ---------------------------------------------------------------------------

API_KEY   = os.getenv("OPENAI_API_KEY")
API_BASE  = os.getenv("API_BASE",    "https://openai.rc.asu.edu/v1")
MODEL     = os.getenv("MODEL_NAME",  "qwen3-30b-a3b-instruct-2507")

_ENDPOINT = f"{API_BASE}/chat/completions"


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def call_model(
    prompt:      str,
    system:      str   = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_tokens:  int   = 1024,
    model:       str   = MODEL,
    timeout:     int   = 60,
    retries:     int   = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    Call the chat completions endpoint and return a result dict:

        {
            "ok":      bool,
            "text":    str | None,    # model's reply, stripped
            "status":  int,           # HTTP status code (-1 on network error)
            "error":   str | None,    # error message if ok=False
        }

    Retries up to `retries` times on 5xx errors or network failures.
    Raises ValueError immediately on bad config (missing key, etc.).
    """
    if not API_KEY or API_KEY == "CREATE FROM Voyager Portal":
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp   = requests.post(_ENDPOINT, headers=headers, json=payload, timeout=timeout, verify=False)
            status = resp.status_code

            if status == 200:
                data = resp.json()
                text = (
                    data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    or ""
                ).strip()
                return {"ok": True, "text": text, "status": status, "error": None}

            # Surface error body for debugging
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text

            last_error = f"HTTP {status}: {err_body}"

            # Retry on server errors; bail immediately on client errors
            if status < 500:
                return {"ok": False, "text": None, "status": status, "error": last_error}

        except requests.RequestException as exc:
            last_error = str(exc)

        if attempt < retries:
            time.sleep(retry_delay * attempt)   # back-off: 2s, 4s

    return {"ok": False, "text": None, "status": -1, "error": last_error}


# ---------------------------------------------------------------------------
# Multi-turn call (for ReAct / Tree-of-Thought loops)
# ---------------------------------------------------------------------------

def call_model_messages(
    messages:    list[dict],
    temperature: float = 0.0,
    max_tokens:  int   = 1024,
    model:       str   = MODEL,
    timeout:     int   = 60,
    retries:     int   = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    Same as call_model() but accepts a raw messages list so callers can
    manage multi-turn history themselves.

    messages format:
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    """
    if not API_KEY or API_KEY == "CREATE FROM Voyager Portal":
        raise ValueError("OPENAI_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp   = requests.post(_ENDPOINT, headers=headers, json=payload, timeout=timeout, verify=False)
            status = resp.status_code

            if status == 200:
                data = resp.json()
                text = (
                    data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    or ""
                ).strip()
                return {"ok": True, "text": text, "status": status, "error": None}

            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text

            last_error = f"HTTP {status}: {err_body}"

            if status < 500:
                return {"ok": False, "text": None, "status": status, "error": last_error}

        except requests.RequestException as exc:
            last_error = str(exc)

        if attempt < retries:
            time.sleep(retry_delay * attempt)

    return {"ok": False, "text": None, "status": -1, "error": last_error}


# ---------------------------------------------------------------------------
# Smoke test — run this file directly to verify your key + VPN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Endpoint : {_ENDPOINT}")
    print(f"Model    : {MODEL}")
    print(f"Key set  : {'yes' if API_KEY else 'NO — set OPENAI_API_KEY'}\n")

    result = call_model(
        prompt="What is 12 + 31? Reply with just the number.",
        system="Reply with only the final answer.",
    )

    if result["ok"]:
        print(f"✅ Success! Model says: {result['text']}")
    else:
        print(f"❌ Failed (HTTP {result['status']}): {result['error']}")
