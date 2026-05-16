#!/usr/bin/env python3
"""Run MMLU benchmark against a local LLM server (OpenAI-compatible API)."""

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_DATA_DIR = os.path.expanduser("~/.cache/mmlu")
DEFAULT_SUBJECT = "abstract_algebra"  # ~100 questions, ~5 min on a typical local model
DEFAULT_TIMEOUT = 60
DEFAULT_NUM_SHOTS = 5
DATA_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

CHOICES = ["A", "B", "C", "D"]


def ensure_data(data_dir):
    """Download and extract MMLU dataset if not already cached."""
    sentinel = os.path.join(data_dir, "test", "abstract_algebra_test.csv")
    if os.path.exists(sentinel):
        return

    print(f"Downloading MMLU dataset from {DATA_URL}...")
    os.makedirs(data_dir, exist_ok=True)
    try:
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "llm-tests"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            tar_bytes = resp.read()
    except urllib.error.URLError as e:
        sys.exit(f"Download failed: {e}. Manually download {DATA_URL} and extract "
                 f"into {data_dir} (resulting in {data_dir}/test/, {data_dir}/dev/).")

    print(f"Extracting {len(tar_bytes):,} bytes...")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
        for member in tar.getmembers():
            if member.name.startswith("data/"):
                member.name = member.name[len("data/"):]
            if not member.name:
                continue
            tar.extract(member, data_dir)

    if not os.path.exists(sentinel):
        sys.exit(f"Tarball extracted but {sentinel} missing — unexpected layout.")
    print(f"Cached at {data_dir}")


def load_csv(path):
    """Load an MMLU CSV. Returns list of (question, [A,B,C,D], answer_letter)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            question, a, b, c, d, answer = row[0], row[1], row[2], row[3], row[4], row[5]
            rows.append((question, [a, b, c, d], answer.strip().upper()))
    return rows


def format_subject(subject):
    return subject.replace("_", " ")


def format_example(question, choices, answer_letter=None):
    parts = [question]
    for letter, choice in zip(CHOICES, choices):
        parts.append(f"{letter}. {choice}")
    parts.append("Answer:")
    s = "\n".join(parts)
    if answer_letter:
        s += f" {answer_letter}\n\n"
    return s


def build_prompt(dev_rows, test_row, subject, num_shots):
    header = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    shots = "".join(format_example(q, ch, ans) for q, ch, ans in dev_rows[:num_shots])
    test_q, test_ch, _ = test_row
    return header + shots + format_example(test_q, test_ch)


def query(model, base_url, prompt, timeout):
    # Raw text completion (not chat) so the model literally continues "Answer: "
    # with a letter token. The chat endpoint frames this as Q&A and the model
    # produces explanation prose instead of a letter — well below random chance
    # once parsing fails. No chat template means no <think> token either.
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 4,
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


def parse_letter(reply):
    """The prompt ends with 'Answer:' — the model's first non-whitespace token
    is the letter answer. Anything else is a parse failure (don't scan deeper:
    that would let stray letters from explanation prose count as answers)."""
    stripped = reply.lstrip()
    if stripped and stripped[0].upper() in CHOICES:
        return stripped[0].upper()
    return None


def evaluate_subject(model, base_url, data_dir, subject, num_shots, limit, timeout):
    dev_path = os.path.join(data_dir, "dev", f"{subject}_dev.csv")
    test_path = os.path.join(data_dir, "test", f"{subject}_test.csv")
    dev_rows = load_csv(dev_path)
    test_rows = load_csv(test_path)
    if limit:
        test_rows = test_rows[:limit]

    n_correct = n_parse_fail = n_errors = 0
    start = time.time()
    last_progress = start

    for i, test_row in enumerate(test_rows):
        prompt = build_prompt(dev_rows, test_row, subject, num_shots)
        gold = test_row[2]
        try:
            result = query(model, base_url, prompt, timeout)
            choice = result["choices"][0]
            reply = choice.get("text") or choice.get("message", {}).get("content", "")
            pred = parse_letter(reply)
            if pred is None:
                n_parse_fail += 1
            elif pred == gold:
                n_correct += 1
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"  [error] {type(e).__name__}: {e}", file=sys.stderr)

        now = time.time()
        if now - last_progress >= 10:
            done = i + 1
            acc = n_correct / done * 100
            rate = done / (now - start)
            print(f"  [{subject}] {done}/{len(test_rows)} ({rate:.2f} q/s), "
                  f"acc so far: {acc:.1f}%", flush=True)
            last_progress = now

    elapsed = time.time() - start
    n_total = len(test_rows)
    return {
        "subject": subject,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_parse_fail": n_parse_fail,
        "n_errors": n_errors,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "elapsed": elapsed,
    }


def list_subjects(data_dir):
    test_dir = os.path.join(data_dir, "test")
    return sorted(
        fname[:-len("_test.csv")]
        for fname in os.listdir(test_dir)
        if fname.endswith("_test.csv")
    )


def parse_args():
    p = argparse.ArgumentParser(description="Run MMLU benchmark against a local LLM server.")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL,
                   help=f"Model name (default: {DEFAULT_MODEL})")
    p.add_argument("-u", "--url", default=DEFAULT_BASE_URL,
                   help=f"Base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("-s", "--subject", default=DEFAULT_SUBJECT,
                   help=f"Subject to evaluate (default: {DEFAULT_SUBJECT}). "
                        "Comma-separated list also accepted.")
    p.add_argument("--all", action="store_true",
                   help="Run all 57 subjects (full MMLU, several hours).")
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="Limit per-subject questions (sanity check / quick run).")
    p.add_argument("-k", "--shots", type=int, default=DEFAULT_NUM_SHOTS,
                   help=f"Number of few-shot examples (default: {DEFAULT_NUM_SHOTS})")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"MMLU data cache directory (default: {DEFAULT_DATA_DIR})")
    p.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_data(args.data_dir)

    if args.all:
        subjects = list_subjects(args.data_dir)
    elif "," in args.subject:
        subjects = [s.strip() for s in args.subject.split(",")]
    else:
        subjects = [args.subject]

    print("=" * 60)
    print("MMLU Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Server: {args.url}")
    print(f"Subjects: {len(subjects)}"
          + (f" (first: {subjects[0]})" if len(subjects) > 1 else f" ({subjects[0]})"))
    print(f"Shots: {args.shots}")
    if args.limit:
        print(f"Limit per subject: {args.limit}")
    print("=" * 60)

    results = []
    overall_start = time.time()
    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {subject}")
        r = evaluate_subject(args.model, args.url, args.data_dir,
                              subject, args.shots, args.limit, args.timeout)
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
        print(f"  {'Subject':<40} {'Acc':>7} {'N':>5}")
        print(f"  {'-'*40} {'-'*7} {'-'*5}")
        for r in sorted(results, key=lambda x: -x["accuracy"]):
            print(f"  {r['subject']:<40} {r['accuracy']*100:>6.1f}% {r['n_total']:>5}")
        print(f"\nMacro avg (per-subject mean): {macro_acc*100:.2f}%")
        print(f"Micro avg (all questions):    {micro_acc*100:.2f}%")
        print(f"Total questions:              {total_q}")
        print(f"Parse failures:               {total_pf}")
        print(f"Wall time:                    {overall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
