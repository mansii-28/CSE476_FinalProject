"""
generate_answers.py — Produces the autograder submission file.

Reads cse_476_final_project_test_data.json and writes
cse_476_final_project_answers.json in the exact format the grader expects:
    [{"output": "..."}, {"output": "..."}, ...]

One entry per question, in the same order as the input file.
Answers are capped at 4,999 characters as required.

Usage:
    python generate_answers.py                         # full run
    python generate_answers.py --limit 20              # quick smoke test
    python generate_answers.py --resume                # skip already-done questions

Output file goes to: outputs/cse_476_final_project_answers.json
Copy it to the project root before submitting.
"""

import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from agent import Agent

load_dotenv()

INPUT_PATH      = Path("data/cse_476_final_project_test_data.json")
OUTPUT_PATH     = Path("outputs/cse_476_final_project_answers.json")
CHECKPOINT_PATH = Path("outputs/cse_476_answers_checkpoint.json")
MAX_CHARS       = 4999


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[generate] Loaded {len(data)} questions from {path}")
    return data


def load_checkpoint() -> dict[int, str]:
    """Load any previously completed answers keyed by index."""
    if not CHECKPOINT_PATH.exists():
        return {}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    done = {int(k): v for k, v in raw.items()}
    print(f"[generate] Checkpoint found — {len(done)} questions already answered.")
    return done


def save_checkpoint(done: dict[int, str]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in done.items()}, f, ensure_ascii=False, indent=2)


def validate(questions: list[dict], answers: list[dict]) -> None:
    """Run the same checks as the professor's validate_results()."""
    if len(questions) != len(answers):
        raise ValueError(
            f"Length mismatch: {len(questions)} questions but {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field at index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(f"Non-string output at index {idx}: {type(answer['output'])}")
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} is {len(answer['output'])} chars — exceeds 5000 limit."
            )
    print(f"[generate] Validation passed — {len(answers)} answers look good.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(questions: list[dict], limit: Optional[int], resume: bool, sleep: float) -> list[str]:
    """
    Call the agent on every question and return a list of answer strings
    in the same order as the input.
    """
    agent = Agent()
    done  = load_checkpoint() if resume else {}

    total = len(questions) if limit is None else min(limit, len(questions))

    for i, q in enumerate(questions[:total]):
        if i in done:
            print(f"[{i+1}/{total}] skipping (already done)")
            continue

        question = q.get("input", "")
        print(f"[{i+1}/{total}] {question[:80]!r}...")

        t0 = time.time()
        try:
            prediction, technique = agent.solve(question)   # domain auto-detected
            output = str(prediction).strip()[:MAX_CHARS]
        except Exception as exc:
            print(f"  !! ERROR: {exc}")
            output = ""

        elapsed = round(time.time() - t0, 2)
        print(f"  → [{technique}] {output[:80]!r}  ({elapsed}s)")

        done[i] = output

        # Save checkpoint every 50 questions
        if (i + 1) % 50 == 0:
            save_checkpoint(done)
            print(f"  [checkpoint saved at {i+1}]")

        if sleep > 0:
            time.sleep(sleep)

    # Final checkpoint save
    save_checkpoint(done)

    # Reconstruct ordered list — fill any gaps with empty string
    return [done.get(i, "") for i in range(total)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate autograder submission file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions (for smoke testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint, skipping already-answered questions.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between API calls (default 0.3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "CREATE FROM Voyager Portal":
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    questions = load_questions(INPUT_PATH)
    answers_text = run(questions, limit=args.limit, resume=args.resume, sleep=args.sleep)

    # Build submission format
    answers = [{"output": text} for text in answers_text]

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    print(f"\n[generate] Wrote {len(answers)} answers to {OUTPUT_PATH}")

    # Validate if full run
    if args.limit is None:
        validate(questions, answers)

    print("\n[generate] Done. Submit: outputs/cse_476_final_project_answers.json")


if __name__ == "__main__":
    main()