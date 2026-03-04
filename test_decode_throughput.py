#!/usr/bin/env python3
"""Benchmark decode (token generation) throughput of a local LLM server."""

import argparse
import urllib.request
import urllib.error
import json
import time

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_TOKENS = [256, 1024, 4096]
DEFAULT_TIMEOUT = 3600  # 60 minutes


def test_decode(target_tokens, model, base_url, timeout):
    """Generate target_tokens tokens and measure decode throughput."""

    prompt = (
        f"Write a very long, detailed story about a space explorer. "
        f"Do not stop writing until you have written at least {target_tokens * 2} words. "
        f"Include many characters, locations, and plot twists. Go:"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": target_tokens,
        # Disable thinking mode per Qwen docs
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
        # Try to prevent early stopping
        "ignore_eos": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    label = format_token_count(target_tokens)
    print(f"\n[{label}] Generating {target_tokens:,} tokens...", flush=True)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            elapsed = time.time() - start_time
            result = json.loads(response.read().decode("utf-8"))

            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", 0)

            reply = result["choices"][0]["message"]["content"].strip()
            finish_reason = result["choices"][0].get("finish_reason", "?")

            # Calculate decode throughput
            decode_tps = completion_tokens / elapsed if elapsed > 0 else 0

            hit_target = completion_tokens >= target_tokens * 0.9  # within 90%
            status = "OK" if hit_target else "SHORT"

            print(f"  {status} ({elapsed:.1f}s, finish: {finish_reason})")
            print(f"  Tokens: {prompt_tokens} prompt + {completion_tokens} completion (target: {target_tokens})")
            print(f"  Decode throughput: {decode_tps:.1f} tok/s")
            print(f"  Response preview: {reply[:100]}...")

            return {
                "success": True, "hit_target": hit_target,
                "time": elapsed, "completion_tokens": completion_tokens,
                "decode_tps": decode_tps, "finish_reason": finish_reason,
            }

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


def format_token_count(n):
    """Format token count as human-readable label."""
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark decode (generation) throughput of a local LLM server."
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
        "-n", "--tokens",
        type=str,
        default=None,
        help="Comma-separated list of token counts to generate (default: 256,1024,4096)"
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
    token_sizes = [int(x) for x in args.tokens.split(",")] if args.tokens else DEFAULT_TOKENS

    print("=" * 60)
    print("Decode Throughput Benchmark")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Server: {base_url}")
    print(f"Target sizes: {', '.join(format_token_count(n) for n in token_sizes)} tokens")
    print(f"Timeout: {timeout}s ({timeout // 60} minutes)")
    print("=" * 60)

    results = []

    for target in token_sizes:
        result = test_decode(target, model, base_url, timeout)
        results.append((target, result))

        if not result["success"]:
            print(f"\n[!] Stopping tests - {format_token_count(target)} generation failed")
            break

    # Summary
    successful = [(n, r) for n, r in results if r.get("success")]
    if successful:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"  {'Target':<10} {'Actual':<10} {'Time':<10} {'Decode tok/s':<15} {'Finish'}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*15} {'-'*10}")
        for target, r in successful:
            actual = r.get("completion_tokens", 0)
            print(f"  {format_token_count(target):<10} {actual:<10} {r['time']:<10.1f} {r.get('decode_tps', 0):<15.1f} {r.get('finish_reason', '?')}")


if __name__ == "__main__":
    main()
