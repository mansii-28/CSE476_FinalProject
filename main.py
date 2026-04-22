"""
main.py — Entry point for the inference-time reasoning agent.

Usage:
    python main.py --data data/dev_data.json --output outputs/results.json
    python main.py --data data/test_data.json --output outputs/test_results.json --mode test
    python main.py --data data/dev_data.json --limit 50              # quick debug run
    python main.py --data data/dev_data.json --domain math           # single domain only
    python main.py --data data/dev_data.json --resume outputs/results.json  # skip already-done

Date JSON format:
    {"input": str, "output": str, "domain": str}
    domains: math (300), common_sense (400), coding (100),
             planning (100), future_prediction (100)
Note: test data will NOT have an "output" key.
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from agent import Agent
from evaluate import evaluate_results

load_dotenv()

KNOWN_DOMAINS = {"math", "common_sense", "coding", "planning", "future_prediction"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> list[dict]:
    """Load dev or test data from a JSON file.

    Real schema: [{"input": "...", "output": "...", "domain": "..."}, ...]
    The "output" key is absent in released test data.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    domain_counts = Counter(d.get("domain", "unknown") for d in data)
    print(f"[main] Loaded {len(data)} instances from {path}")
    print(f"[main] Domain breakdown: { {k: domain_counts[k] for k in sorted(domain_counts)} }")
    return data


def load_existing_results(path: str) -> dict[int, dict]:
    """Load a previous results file keyed by instance id, for --resume support."""
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    done = {r["id"]: r for r in results}
    print(f"[main] Resuming — found {len(done)} already-completed results in {path}")
    return done


def save_results(results: list[dict], path: str) -> None:
    """Write results to a JSON file, creating parent dirs if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[main] Saved {len(results)} results to {path}")


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------

def run(
    data: list[dict],
    mode: str = "dev",
    sleep_between: float = 0.3,
    domain_filter: str | None = None,
    existing: dict[int, dict] | None = None,
) -> list[dict]:
    """
    Iterate over instances and call the agent on each one.

    Returns a list of result dicts:
        {
            "id":             int,
            "domain":         str,
            "input":          str,
            "expected":       str | None,   # None in test mode
            "prediction":     str,
            "technique_used": str,
            "elapsed_sec":    float,
            "error":          str | None,
        }
    """
    agent = Agent()
    existing = existing or {}
    results = list(existing.values())  # carry forward any resumed results

    to_run = [
        (i, inst) for i, inst in enumerate(data)
        if i not in existing
        and (domain_filter is None or inst.get("domain") == domain_filter)
    ]

    if domain_filter:
        print(f"[main] Filtering to domain={domain_filter!r} — {len(to_run)} instances to run.")

    skipped = len(data) - len(to_run) - len(existing)
    if skipped > 0:
        print(f"[main] Skipping {skipped} instances (domain filter).")

    for idx, (i, instance) in enumerate(to_run):
        question = instance.get("input", "")
        domain   = instance.get("domain", "unknown")
        expected = instance.get("output")          # key is "output", not "expected_output"
                                                   # will be None for test data

        print(f"[{idx+1}/{len(to_run)}] id={i} domain={domain!r}  {question[:80]!r}...")

        t0 = time.time()
        try:
            prediction, technique = agent.solve(question, domain=domain)
            error = None
        except Exception as exc:
            prediction = ""
            technique  = "error"
            error      = str(exc)
            print(f"  !! ERROR: {exc}")

        elapsed = round(time.time() - t0, 2)
        print(f"  → [{technique}] {prediction[:100]!r}  ({elapsed}s)")

        results.append({
            "id":             i,
            "domain":         domain,
            "input":          question,
            "expected":       expected,
            "prediction":     prediction,
            "technique_used": technique,
            "elapsed_sec":    elapsed,
            "error":          error,
        })

        if sleep_between > 0:
            time.sleep(sleep_between)

    # Always return sorted by id so the output file is stable
    results.sort(key=lambda r: r["id"])
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_domain_summary(results: list[dict]) -> None:
    """Print per-domain technique usage and error counts."""
    by_domain: dict[str, list] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    print("\n[main] ── Domain summary ──────────────────────────────")
    for domain in sorted(by_domain):
        rows        = by_domain[domain]
        errors      = sum(1 for r in rows if r["error"])
        techniques  = Counter(r["technique_used"] for r in rows)
        avg_elapsed = round(sum(r["elapsed_sec"] for r in rows) / len(rows), 2)
        print(f"  {domain:<20} n={len(rows)}  errors={errors}  avg={avg_elapsed}s")
        for tech, count in techniques.most_common():
            print(f"    {tech:<30} {count}x")
    print("[main] ──────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the inference-time reasoning agent.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/dev_data.json",
        help="Path to input data (JSON list of instances).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results.json",
        help="Path to write output results.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="'dev' evaluates after inference (labels required); 'test' skips evaluation.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=sorted(KNOWN_DOMAINS),
        help="Only run instances from this domain (useful for targeted debugging).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run on the first N instances after filtering.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between API calls (default 0.3).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RESULTS_JSON",
        help="Path to a previous results file; already-completed ids will be skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "CREATE FROM Voyager Portal":
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "Export it or add it to a .env file:\n"
            "  export OPENAI_API_KEY=your_key_here"
        )

    # Load data
    data = load_data(args.data)

    # Apply --limit BEFORE passing to run(), after any domain filter
    if args.domain:
        filtered = [d for d in data if d.get("domain") == args.domain]
    else:
        filtered = data

    if args.limit:
        filtered = filtered[: args.limit]
        print(f"[main] Limiting to first {args.limit} instances (after domain filter).")

    # Re-index so ids match original positions in full data
    # (we use enumerate index from full data inside run(), so pass full data + let run filter)
    existing = load_existing_results(args.resume) if args.resume else {}

    print(f"\n[main] Starting agent in '{args.mode}' mode on {len(filtered)} instances...\n")

    results = run(
        data          = data if not args.limit else filtered,
        mode          = args.mode,
        sleep_between = args.sleep,
        domain_filter = args.domain,
        existing      = existing,
    )

    # Save results
    save_results(results, args.output)

    # Per-domain technique breakdown
    print_domain_summary(results)

    # Evaluate accuracy (dev mode only)
    if args.mode == "dev":
        print("[main] Running evaluation...\n")
        evaluate_results(results)

    print("[main] Done.")


if __name__ == "__main__":
    main()
