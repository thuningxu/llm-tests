#!/usr/bin/env python3
"""Probe the actual supported context window size of a local LM Studio model."""

import argparse
import urllib.request
import urllib.error
import json
import time
import secrets

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_TEST_SIZES = [1, 16, 128, 192, 256]
DEFAULT_TIMEOUT = 3600  # 60 minutes
SECRET_INTERVAL_K = 16  # place a secret every N thousand tokens


def _filler_text(num_tokens):
    """Generate repetitive text to fill context. ~4 chars per token estimate."""
    chunk = "The quick brown fox jumps over the lazy dog. "
    chars_needed = num_tokens * 4
    repetitions = chars_needed // len(chunk) + 1
    return (chunk * repetitions)[:chars_needed]


def _count_tokens(text):
    """Rough token count estimate (~4 chars per token for English)."""
    return len(text) // 4


def build_prompt_with_secrets(target_k):
    """Build a prompt with secret codes distributed every SECRET_INTERVAL_K tokens.

    Returns (user_message, list_of_secret_codes).
    Each secret is placed at ~0K, ~16K, ~32K, ... positions so we can detect
    middle-collapse where the server silently drops context from the center."""
    target_tokens = target_k * 1000
    interval = SECRET_INTERVAL_K * 1000

    # How many secret slots fit in this context?
    num_secrets = max(1, (target_tokens // interval) + 1)

    secret_codes = []
    parts = []

    for i in range(num_secrets):
        # Random 12-char hex code — unpredictable, can't be guessed from position
        code = f"CODE-{secrets.token_hex(6).upper()}"
        secret_codes.append(code)

        if i == 0:
            # First secret near the top — minimal filler before it
            parts.append(f"The secret code is {code}.\n")
        else:
            # Fill ~interval tokens of filler between secrets
            filler = _filler_text(interval - 50)  # -50 for the secret line itself
            parts.append(filler)
            parts.append(f"The secret code is {code}.\n")

    user_message = (
        "I'm giving you text with hidden codes. List ALL of them.\n\n"
        "START:\n"
        + "".join(parts)
        + "\nEND.\n\n"
        "List every secret code you find, one per line."
    )

    return user_message, secret_codes


def test_context_size(target_k, model, base_url, timeout):
    """Test if the model can handle a context of approximately target_k thousand tokens.

    Places multiple unique secret codes throughout the prompt (every 16K tokens)
    and checks recall for each one independently to detect middle-collapse."""
    user_message, secrets = build_prompt_with_secrets(target_k)

    # Scale max_tokens based on number of secrets so the model has room to list them all
    max_tokens = max(4096, len(secrets) * 50 + 1024)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_message},
            # Prefill an empty thinking block to suppress CoT.
            # The chat_template_kwargs/enable_thinking knob is unreliable across servers;
            # this prefill forces the model to skip CoT regardless of template config.
            {"role": "assistant", "content": "\x3cthinking\x3e\n\n\x3c/thinking\x3e\n\n"},
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": max_tokens,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    estimated_tokens = _count_tokens(user_message)
    print(f"\n[{target_k}K] Testing ~{estimated_tokens:,} tokens... ({len(secrets)} secrets every {SECRET_INTERVAL_K}K)", flush=True)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            elapsed = time.time() - start_time
            result = json.loads(response.read().decode("utf-8"))

            usage = result.get("usage", {})
            actual_prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")

            reply = result["choices"][0]["message"]["content"].strip()

            # Check recall for each secret independently
            reply_upper = reply.upper().replace(" ", "-")
            results_per_secret = []
            all_found = True
            for code in secrets:
                found = code.upper() in reply_upper
                results_per_secret.append(found)
                if not found:
                    all_found = False

            # Calculate throughput
            prompt_tps = actual_prompt_tokens / elapsed if elapsed > 0 else 0
            completion_tps = completion_tokens / elapsed if elapsed > 0 else 0

            status = "OK" if all_found else "PARTIAL"
            found_count = sum(results_per_secret)
            print(f"  {status} ({elapsed:.1f}s) -- recalled {found_count}/{len(secrets)} secrets")
            print(f"  Tokens: {actual_prompt_tokens} prompt + {completion_tokens} completion")
            print(f"  Throughput: {prompt_tps:.1f} prompt tok/s, {completion_tps:.1f} completion tok/s")

            # Show per-secret recall map
            if len(secrets) > 1:
                markers = []
                for i, found in enumerate(results_per_secret):
                    pos_k = i * SECRET_INTERVAL_K
                    marker = "OK" if found else "MISS"
                    markers.append(f"[{pos_k}K]={marker}")
                print(f"  Recall map: {' '.join(markers)}")

            # Show response preview (truncated for large responses)
            preview_len = min(300, len(reply))
            suffix = "..." if len(reply) > preview_len else ""
            print(f"  Response ({len(reply)} chars): {reply[:preview_len]}{suffix}")

            return {"success": True, "recall_all": all_found,
                    "recall_partial": found_count / len(secrets),
                    "secrets_total": len(secrets), "secrets_found": found_count,
                    "time": elapsed,
                    "prompt_tokens": actual_prompt_tokens, "completion_tokens": completion_tokens,
                    "prompt_tps": prompt_tps, "completion_tps": completion_tps}

    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        error_body = e.read().decode("utf-8")
        print(f"  FAILED ({e.code}) after {elapsed:.1f}s")
        try:
            error_json = json.loads(error_body)
            print(f"  Error: {error_json.get('error', {}).get('message', error_body)[:200]}")
        except Exception:
            print(f"  Error: {error_body[:200]}")
        return {"success": False, "error": e.code}

    except urllib.error.URLError as e:
        elapsed = time.time() - start_time
        print(f"  FAILED (connection error: {e.reason}) after {elapsed:.1f}s")
        return {"success": False, "error": str(e.reason)}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  TIMEOUT/ERROR after {elapsed:.1f}s: {type(e).__name__}")
        return {"success": False, "error": str(e)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe the context window size of a local LLM server."
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to test (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-u", "--url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API server (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "-s", "--sizes",
        type=str,
        default=None,
        help="Comma-separated list of context sizes in K to test (default: 1,16,128,192,256)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds per request (default: {DEFAULT_TIMEOUT})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = args.model
    base_url = args.url
    timeout = args.timeout
    test_sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else DEFAULT_TEST_SIZES

    print("=" * 60)
    print("Context Window Size Test")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Server: {base_url}")
    print(f"Testing sizes: {', '.join(f'{k}K' for k in test_sizes)}")
    print(f"Timeout: {timeout}s ({timeout // 60} minutes)")
    print(f"Secret interval: every {SECRET_INTERVAL_K}K tokens (catches middle-collapse)")
    print("Thinking mode: DISABLED via empty thinking prefill")
    print("=" * 60)

    results = []
    max_working = 0
    max_with_recall = 0

    for i, size_k in enumerate(test_sizes):
        result = test_context_size(size_k, model, base_url, timeout)
        results.append((size_k, result))

        # First test (smallest size) is validation - must pass with full recall
        if i == 0:
            if not result["success"] or not result.get("recall_all"):
                print("\n" + "=" * 60)
                print("ABORT: baseline test failed!")
                print("=" * 60)
                print("The first test must pass with full recall to validate the setup.")
                print("Possible issues:")
                print("  - Model not loaded or server not running")
                print("  - Model doesn't support the expected prompt format")
                print("  - max_tokens too low to capture the answer")
                return

        if result["success"]:
            max_working = size_k
            if result.get("recall_all"):
                max_with_recall = size_k
        else:
            print(f"\n[!] Stopping tests - {size_k}K context failed")
            break

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Max context that worked:        {max_working}K tokens")
    print(f"Max context with full recall:   {max_with_recall}K tokens")

    # Throughput summary
    successful_results = [(k, r) for k, r in results if r.get("success")]
    if successful_results:
        print("\nThroughput by context size:")
        print(f"  {'Size':<8} {'Recall':<10} {'Time':<10} {'Prompt tok/s':<15} {'Completion tok/s'}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*15} {'-'*15}")
        for size_k, r in successful_results:
            recall_str = f"{r.get('secrets_found', '?')}/{r.get('secrets_total', '?')}"
            print(f"  {size_k}K{'':<4} ({recall_str}) {r['time']:<10.1f} {r.get('prompt_tps', 0):<15.1f} {r.get('completion_tps', 0):.1f}")

    if max_with_recall >= 256:
        print("\nFull 256K context supported with recall!")
    elif max_with_recall > 0:
        print(f"\nYour system supports up to ~{max_with_recall}K context.")


if __name__ == "__main__":
    main()
