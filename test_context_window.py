#!/usr/bin/env python3
"""Probe the actual supported context window size of a local LM Studio model."""

import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3.5-35b-a3b"

# Test sizes: 1K to validate, then large contexts
TEST_SIZES_K = [1, 128, 192, 256]

# 30 minute timeout
TIMEOUT_SECONDS = 1800


def generate_filler_text(target_tokens):
    """Generate repetitive text to fill context. ~4 chars per token estimate."""
    chunk = "The quick brown fox jumps over the lazy dog. "
    chars_needed = target_tokens * 4
    repetitions = chars_needed // len(chunk) + 1
    text = (chunk * repetitions)[:chars_needed]
    return text


def count_tokens_estimate(text):
    """Rough token count estimate (~4 chars per token for English)."""
    return len(text) // 4


def test_context_size(target_k):
    """Test if the model can handle a context of approximately target_k thousand tokens."""
    target_tokens = target_k * 1000

    filler_tokens = target_tokens - 200
    filler_text = generate_filler_text(filler_tokens)

    user_message = f"""I'm giving you text with a hidden code. Find it.

START:
The secret code is BLUE-ELEPHANT-42.
{filler_text}
END.

What is the secret code? Give me ONLY the code as your final answer."""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 100,
        # Disable thinking mode per Qwen docs
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    estimated_tokens = count_tokens_estimate(user_message)
    print(f"\n[{target_k}K] Testing ~{estimated_tokens:,} tokens...", flush=True)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            elapsed = time.time() - start_time
            result = json.loads(response.read().decode("utf-8"))

            usage = result.get("usage", {})
            actual_prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")

            reply = result["choices"][0]["message"]["content"].strip()

            # Check if model found the secret code anywhere in response
            found_code = "BLUE-ELEPHANT-42" in reply.upper().replace(" ", "-")

            # Calculate throughput
            prompt_tps = actual_prompt_tokens / elapsed if elapsed > 0 else 0
            # Estimate time spent on completion (rough: assume linear with tokens)
            total_tokens = actual_prompt_tokens + completion_tokens
            completion_tps = completion_tokens / elapsed if elapsed > 0 else 0

            status = "OK" if found_code else "RECALL FAILED"
            print(f"  {status} ({elapsed:.1f}s)")
            print(f"  Tokens: {actual_prompt_tokens} prompt + {completion_tokens} completion")
            print(f"  Throughput: {prompt_tps:.1f} prompt tok/s, {completion_tps:.1f} completion tok/s")
            print(f"  Response: {reply[:200]}")

            return {"success": True, "recall": found_code, "time": elapsed,
                    "prompt_tokens": actual_prompt_tokens, "completion_tokens": completion_tokens,
                    "prompt_tps": prompt_tps, "completion_tps": completion_tps}

    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        error_body = e.read().decode("utf-8")
        print(f"  FAILED ({e.code}) after {elapsed:.1f}s")
        try:
            error_json = json.loads(error_body)
            print(f"  Error: {error_json.get('error', {}).get('message', error_body)[:200]}")
        except:
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


def main():
    print("=" * 60)
    print("Context Window Size Test")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Testing sizes: {', '.join(f'{k}K' for k in TEST_SIZES_K)}")
    print(f"Timeout: {TIMEOUT_SECONDS}s ({TIMEOUT_SECONDS // 60} minutes)")
    print("Thinking mode: DISABLED via chat_template_kwargs")
    print("=" * 60)

    results = []
    max_working = 0
    max_with_recall = 0

    for i, size_k in enumerate(TEST_SIZES_K):
        result = test_context_size(size_k)
        results.append((size_k, result))

        # First test (1K) is validation - must pass with recall
        if i == 0:
            if not result["success"] or not result.get("recall"):
                print("\n" + "=" * 60)
                print("ABORT: 1K baseline test failed!")
                print("=" * 60)
                print("The 1K test must pass with recall to validate the test setup.")
                print("Possible issues:")
                print("  - Model not loaded or server not running")
                print("  - Model doesn't support the expected prompt format")
                print("  - max_tokens too low to capture the answer")
                return

        if result["success"]:
            max_working = size_k
            if result.get("recall"):
                max_with_recall = size_k
        else:
            print(f"\n[!] Stopping tests - {size_k}K context failed")
            break

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Max context that worked:        {max_working}K tokens")
    print(f"Max context with recall intact: {max_with_recall}K tokens")

    # Throughput summary
    successful_results = [(k, r) for k, r in results if r.get("success") and r.get("recall")]
    if successful_results:
        print("\nThroughput by context size:")
        print(f"  {'Size':<8} {'Time':<10} {'Prompt tok/s':<15} {'Completion tok/s'}")
        print(f"  {'-'*8} {'-'*10} {'-'*15} {'-'*15}")
        for size_k, r in successful_results:
            print(f"  {size_k}K{'':<5} {r['time']:<10.1f} {r.get('prompt_tps', 0):<15.1f} {r.get('completion_tps', 0):.1f}")

    if max_with_recall >= 256:
        print("\nFull 256K context supported with recall!")
    elif max_with_recall > 0:
        print(f"\nYour system supports up to ~{max_with_recall}K context.")


if __name__ == "__main__":
    main()
