#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.0",
# ]
# ///
"""Run HumanEval (OpenAI) against a local LLM server (OpenAI-compatible API).

164 Python function-completion problems. Each prompt is a function signature
+ docstring; the model continues with the function body. We then execute the
combined code + hidden tests in a subprocess (timeout-capped) and count
pass@1.

Run with:
    uv run test_humaneval.py [args]

Security: model output is executed locally. Subprocess timeout (default 10s)
and Python -I (isolated mode) provide minimal sandboxing. Do NOT point this
at an untrusted/adversarial model.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_TIMEOUT = 300
DEFAULT_EXEC_TIMEOUT = 10
DEFAULT_MAX_TOKENS = 512

DATASET = "openai_humaneval"

# Stop sequences truncate generation at logical boundaries — most models
# happily keep generating example usage or extra functions otherwise.
STOP_SEQUENCES = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```"]


def load_problems():
    print(f"Loading {DATASET}...", flush=True)
    ds = load_dataset(DATASET, split="test")
    print(f"  {len(ds):,} problems", flush=True)
    return list(ds)


def query(model, base_url, prompt, timeout, max_tokens):
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stop": STOP_SEQUENCES,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def clean_completion(text):
    """Strip markdown fences and trim trailing junk after the function body."""
    text = re.sub(r"^```(?:python)?\n", "", text)
    text = re.sub(r"\n```.*$", "", text, flags=re.DOTALL)
    return text


def run_test(prompt, completion, test_code, entry_point, exec_timeout):
    """Execute the candidate code + hidden tests in a subprocess.
    Returns (passed, error_message)."""
    full_code = (
        prompt + completion + "\n\n"
        + test_code + "\n\n"
        + f"check({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True, text=True,
            timeout=exec_timeout,
        )
        if result.returncode == 0:
            return True, None
        err = (result.stderr or result.stdout or "").strip().splitlines()
        return False, err[-1] if err else f"exit {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"timeout ({exec_timeout}s)"
    finally:
        import os
        try:
            os.unlink(path)
        except OSError:
            pass


def evaluate(model, base_url, problems, limit, jobs, timeout, exec_timeout, max_tokens):
    if limit:
        problems = problems[:limit]

    n_pass = n_fail = n_errors = 0
    fail_samples = []  # (task_id, error_message) — for first few failures
    start = time.time()
    last_progress = start

    def run_one(prob):
        result = query(model, base_url, prob["prompt"], timeout, max_tokens)
        text = result["choices"][0].get("text", "")
        completion = clean_completion(text)
        passed, err = run_test(prob["prompt"], completion, prob["test"],
                                prob["entry_point"], exec_timeout)
        return prob["task_id"], passed, err

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(run_one, p) for p in problems]
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                task_id, passed, err = fut.result()
                if passed:
                    n_pass += 1
                else:
                    n_fail += 1
                    if len(fail_samples) < 5:
                        fail_samples.append((task_id, err))
            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    print(f"  [error] {type(e).__name__}: {e}", file=sys.stderr)

            now = time.time()
            if now - last_progress >= 10:
                pass_rate = n_pass / done * 100
                rate = done / (now - start)
                print(f"  {done}/{len(problems)} ({rate:.2f} q/s), "
                      f"pass@1: {pass_rate:.1f}% (errors: {n_errors})", flush=True)
                last_progress = now

    elapsed = time.time() - start
    n_total = len(problems)
    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_errors": n_errors,
        "pass_at_1": n_pass / n_total if n_total else 0.0,
        "elapsed": elapsed,
        "fail_samples": fail_samples,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run HumanEval pass@1 against a local LLM server.")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL)
    p.add_argument("-u", "--url", default=DEFAULT_BASE_URL)
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="Limit number of problems. First-run sanity: -n 5.")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="Concurrent in-flight requests (default: 1).")
    p.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"Per-request HTTP timeout (default: {DEFAULT_TIMEOUT}s)")
    p.add_argument("--exec-timeout", type=int, default=DEFAULT_EXEC_TIMEOUT,
                   help=f"Per-problem code-execution timeout (default: {DEFAULT_EXEC_TIMEOUT}s)")
    return p.parse_args()


def main():
    args = parse_args()
    problems = load_problems()

    print("=" * 60)
    print("HumanEval (pass@1)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Server: {args.url}")
    print(f"Problems: {len(problems) if not args.limit else args.limit}")
    print(f"Jobs: {args.jobs}, max_tokens: {args.max_tokens}")
    print("=" * 60)

    r = evaluate(args.model, args.url, problems, args.limit, args.jobs,
                 args.timeout, args.exec_timeout, args.max_tokens)

    print(f"\nResult: {r['n_pass']}/{r['n_total']} = {r['pass_at_1']*100:.1f}% pass@1")
    print(f"  failures: {r['n_fail']}  errors: {r['n_errors']}  wall: {r['elapsed']:.1f}s")

    if r["fail_samples"]:
        print("\nSample failures:")
        for task_id, err in r["fail_samples"]:
            print(f"  {task_id}: {err}")


if __name__ == "__main__":
    main()
