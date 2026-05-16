#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.0",
# ]
# ///
"""Run MMLU-Pro benchmark against a local LLM server (OpenAI-compatible API).

MMLU-Pro (TIGER-Lab, 2024) is the harder successor to MMLU: 12k questions,
10 answer choices per question (random chance 10% vs 25%), and questions
designed to require multi-step reasoning. Canonical eval is CoT-enabled.

Run with:
    uv run test_mmlu_pro.py [args]

uv auto-creates an isolated venv with the `datasets` dep declared above.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_CATEGORY = "history"  # smallest category at 381 questions
DEFAULT_TIMEOUT = 300
DEFAULT_NUM_SHOTS = 5
DEFAULT_MAX_TOKENS = 2048

CHOICES = list("ABCDEFGHIJ")
DATASET = "TIGER-Lab/MMLU-Pro"
ANSWER_PATTERN = re.compile(r"answer\s+is\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE)


def load_splits():
    """Load test + validation via HF datasets (cached in ~/.cache/huggingface)."""
    print(f"Loading {DATASET} (first run downloads ~5MB)...", flush=True)
    test = load_dataset(DATASET, split="test")
    val = load_dataset(DATASET, split="validation")
    print(f"  test: {len(test):,} rows  validation: {len(val):,} rows", flush=True)
    return test, val


def format_example(row, include_answer=True):
    """Format one row as a CoT example. If include_answer, include the gold
    CoT reasoning + final answer (for few-shot examples). Otherwise leave the
    answer open for the model to generate."""
    parts = ["Question:", row["question"], "Options:"]
    for i, opt in enumerate(row["options"]):
        parts.append(f"{CHOICES[i]}. {opt}")
    if include_answer:
        cot = row.get("cot_content", "").strip()
        cot = re.sub(r"^A:\s*", "Answer: ", cot)
        parts.append(cot + "\n")
    else:
        parts.append("Answer: Let's think step by step.")
    return "\n".join(parts)


def build_prompt(few_shot_rows, test_row, category, num_shots):
    header = (f"The following are multiple choice questions (with answers) about "
              f"{category}. Think step by step and then finish your answer with "
              f'"the answer is (X)" where X is the correct letter choice.\n\n')
    shots = "\n\n".join(format_example(r, include_answer=True)
                        for r in few_shot_rows[:num_shots])
    test_part = format_example(test_row, include_answer=False)
    return header + shots + "\n\n" + test_part


def query(model, base_url, prompt, timeout, max_tokens):
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        # MMLU-Pro is CoT-enabled — let the model reason. We parse the final
        # "the answer is (X)" pattern from anywhere in the response.
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


def parse_answer(reply):
    """Find the LAST 'the answer is (X)' match in the reply. Returns None if
    no answer pattern is found (parse failure)."""
    matches = ANSWER_PATTERN.findall(reply)
    if matches:
        return matches[-1].upper()
    return None


def filter_category(rows, category):
    return [r for r in rows if r["category"] == category]


def evaluate_category(model, base_url, test_rows, val_rows, category, num_shots,
                      limit, timeout, max_tokens, jobs):
    cat_test = filter_category(test_rows, category)
    cat_val = filter_category(val_rows, category)
    if not cat_test:
        cats = sorted({r["category"] for r in test_rows})
        sys.exit(f"No test rows for category {category!r}. Available: {cats}")
    if limit:
        cat_test = cat_test[:limit]

    n_correct = n_parse_fail = n_errors = 0
    start = time.time()
    last_progress = start

    def run_one(row):
        prompt = build_prompt(cat_val, row, category, num_shots)
        gold = row["answer"].strip().upper()
        result = query(model, base_url, prompt, timeout, max_tokens)
        choice = result["choices"][0]
        reply = choice.get("text") or choice.get("message", {}).get("content", "")
        pred = parse_answer(reply)
        if pred is None:
            return "parse_fail"
        return "correct" if pred == gold else "wrong"

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(run_one, row) for row in cat_test]
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                outcome = fut.result()
                if outcome == "correct":
                    n_correct += 1
                elif outcome == "parse_fail":
                    n_parse_fail += 1
            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    print(f"  [error] {type(e).__name__}: {e}", file=sys.stderr)

            now = time.time()
            if now - last_progress >= 10:
                acc = n_correct / done * 100
                rate = done / (now - start)
                print(f"  [{category}] {done}/{len(cat_test)} ({rate:.2f} q/s), "
                      f"acc so far: {acc:.1f}% (parse-fail: {n_parse_fail})", flush=True)
                last_progress = now

    elapsed = time.time() - start
    n_total = len(cat_test)
    return {
        "category": category,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_parse_fail": n_parse_fail,
        "n_errors": n_errors,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "elapsed": elapsed,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run MMLU-Pro benchmark against a local LLM server.")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL,
                   help=f"Model name (default: {DEFAULT_MODEL})")
    p.add_argument("-u", "--url", default=DEFAULT_BASE_URL,
                   help=f"Base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("-c", "--category", default=DEFAULT_CATEGORY,
                   help=f"Category to evaluate (default: {DEFAULT_CATEGORY}). "
                        "Comma-separated list also accepted.")
    p.add_argument("--all", action="store_true",
                   help="Run all 14 categories (full MMLU-Pro, many hours).")
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="Limit per-category questions. Recommended for first run: -n 20.")
    p.add_argument("-k", "--shots", type=int, default=DEFAULT_NUM_SHOTS,
                   help=f"Number of few-shot examples (default: {DEFAULT_NUM_SHOTS})")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"Max generation tokens for CoT (default: {DEFAULT_MAX_TOKENS})")
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="Concurrent in-flight requests (default: 1). "
                        "Speedup depends on server-side continuous batching.")
    p.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    return p.parse_args()


def main():
    args = parse_args()
    test_rows, val_rows = load_splits()

    if args.all:
        categories = sorted({r["category"] for r in test_rows})
    elif "," in args.category:
        categories = [c.strip() for c in args.category.split(",")]
    else:
        categories = [args.category]

    print("=" * 60)
    print("MMLU-Pro Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Server: {args.url}")
    print(f"Categories: {len(categories)}"
          + (f" (first: {categories[0]})" if len(categories) > 1 else f" ({categories[0]})"))
    print(f"Shots: {args.shots}, max_tokens: {args.max_tokens}, jobs: {args.jobs}")
    if args.limit:
        print(f"Limit per category: {args.limit}")
    print("=" * 60)

    results = []
    overall_start = time.time()
    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] {category}")
        r = evaluate_category(args.model, args.url, test_rows, val_rows,
                              category, args.shots, args.limit,
                              args.timeout, args.max_tokens, args.jobs)
        results.append(r)
        print(f"  -> {r['n_correct']}/{r['n_total']} = {r['accuracy']*100:.1f}%"
              f"  (parse-fail: {r['n_parse_fail']}, errors: {r['n_errors']},"
              f" {r['elapsed']:.1f}s)", flush=True)

    overall_elapsed = time.time() - overall_start

    if len(results) > 1:
        total_q = sum(r["n_total"] for r in results)
        total_correct = sum(r["n_correct"] for r in results)
        total_pf = sum(r["n_parse_fail"] for r in results)
        macro_acc = sum(r["accuracy"] for r in results) / len(results)
        micro_acc = total_correct / total_q if total_q else 0
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"  {'Category':<25} {'Acc':>7} {'N':>5}")
        print(f"  {'-'*25} {'-'*7} {'-'*5}")
        for r in sorted(results, key=lambda x: -x["accuracy"]):
            print(f"  {r['category']:<25} {r['accuracy']*100:>6.1f}% {r['n_total']:>5}")
        print(f"\nMacro avg (per-category mean): {macro_acc*100:.2f}%")
        print(f"Micro avg (all questions):     {micro_acc*100:.2f}%")
        print(f"Total questions:               {total_q}")
        print(f"Parse failures:                {total_pf}")
        print(f"Wall time:                     {overall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
