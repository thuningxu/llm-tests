#!/usr/bin/env python3
"""Simple script to test LM Studio's OpenAI-compatible API (no dependencies)."""

import urllib.request
import urllib.error
import json

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3.5-35b-a3b"


def test_chat_completion():
    """Test the chat completions endpoint."""
    url = f"{BASE_URL}/chat/completions"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2? Answer briefly."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print(f"Sending request to {url}...")
    print(f"Model: {MODEL}\n")

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            print("Response received!")
            print("-" * 40)
            print(result["choices"][0]["message"]["content"])
            print("-" * 40)
            print(f"\nUsage: {result.get('usage', 'N/A')}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        print(e.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Connection Error: {e.reason}")
        print("Make sure LM Studio server is running on 127.0.0.1:1234")


def list_models():
    """List available models."""
    url = f"{BASE_URL}/models"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            models = json.loads(response.read().decode("utf-8"))
            print("Available models:")
            for model in models.get("data", []):
                print(f"  - {model['id']}")
    except urllib.error.URLError as e:
        print(f"Error listing models: {e.reason}")


if __name__ == "__main__":
    print("Testing LM Studio API\n")
    print("=" * 40)

    print("\n1. Listing models...")
    list_models()

    print("\n2. Testing chat completion...")
    test_chat_completion()
