# llm-tests

Simple tools for testing local LLM servers (LM Studio, etc.) via OpenAI-compatible APIs.

## Scripts

### test_lm_studio.py

Basic connectivity test — lists available models and sends a simple chat completion request.

```bash
python3 test_lm_studio.py
```

### test_context_window.py

Probes the maximum usable context window size by sending increasingly large prompts with a hidden "secret code", then checking if the model can recall it.

```bash
# Test with default model and sizes (1K, 16K, 128K, 192K, 256K)
python3 test_context_window.py

# Test a specific model
python3 test_context_window.py --model "qwen3.5-27b"

# Test specific context sizes
python3 test_context_window.py --model "qwen3.5-27b" --sizes "1,16,64"

# Custom server URL and timeout
python3 test_context_window.py --url "http://localhost:8080/v1" --timeout 7200
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--model` | Model name to test | `qwen/qwen3.5-35b-a3b` |
| `-u`, `--url` | Base URL of the API server | `http://127.0.0.1:1234/v1` |
| `-s`, `--sizes` | Comma-separated context sizes in K | `1,16,128,192,256` |
| `-t`, `--timeout` | Timeout in seconds per request | `3600` (60 min) |

#### How it works

1. Generates a prompt with a secret code (`BLUE-ELEPHANT-42`) followed by filler text
2. Asks the model to recall the code
3. Reports success/failure, timing, and token throughput (prompt tok/s, completion tok/s)
4. The first test (smallest size) serves as a baseline validation — if it fails, the suite aborts

#### Output

```
[64K] Testing ~63,842 tokens...
  OK (199.6s)
  Tokens: 56773 prompt + 1104 completion
  Throughput: 284.5 prompt tok/s, 5.5 completion tok/s
```

### test_decode_throughput.py

Benchmarks decode (token generation) throughput by requesting the model to generate a configurable number of tokens. Uses the chat endpoint with an empty-`<think>` assistant prefill to suppress chain-of-thought and prevent early termination on Qwen3-family models, so the run reliably fills `max_tokens`.

```bash
# Test with default sizes (256, 1024, 4096 tokens)
python3 test_decode_throughput.py

# Test a specific model with custom token counts
python3 test_decode_throughput.py --model "qwen3.5-9b" --tokens "256,8192,32768"

# Test against a remote server
python3 test_decode_throughput.py --model "qwen3.5-9b" --url "http://10.0.0.217:1234/v1" --tokens "256,32768"
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--model` | Model name to test | `qwen/qwen3.5-35b-a3b` |
| `-u`, `--url` | Base URL of the API server | `http://127.0.0.1:1234/v1` |
| `-n`, `--tokens` | Comma-separated token counts to generate | `256,1024,4096` |
| `-t`, `--timeout` | Timeout in seconds per request | `3600` (60 min) |

#### Output

```
[32K] Generating 32,768 tokens...
  OK (503.5s, finish: length)
  Tokens: 46 prompt + 32768 completion (target: 32768)
  Decode throughput: 65.1 tok/s
```

### test_mmlu.py

Runs the classic [MMLU](https://github.com/hendrycks/test) benchmark (57 subjects, 4-option multiple choice). Uses `/v1/completions` so the model literally continues `Answer:` with a letter — the canonical eval format. Auto-downloads dataset to `~/.cache/mmlu/` on first run.

```bash
# One subject (~80s on a 27B at ~1.2 q/s)
python3 test_mmlu.py -m qwen3.6-27b -u http://10.0.0.130:1234/v1

# Quick sanity check
python3 test_mmlu.py -m qwen3.6-27b --limit 5

# All 57 subjects (several hours)
python3 test_mmlu.py -m qwen3.6-27b --all
```

### test_mmlu_pro.py

Runs [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (14 categories, 10-option, reasoning-heavy). CoT-enabled by default; parses `the answer is (X)` from the response. Uses the `datasets` library — invoke via `uv run` so the dep is auto-installed in an ephemeral venv:

```bash
# Default: one category (history, 381 questions)
uv run test_mmlu_pro.py -m qwen3.6-27b -u http://10.0.0.130:1234/v1 --limit 20

# All 14 categories (~hours-to-days depending on model speed)
uv run test_mmlu_pro.py -m qwen3.6-27b --all --limit 50
```

A 27B-class model can take ~60-90s per MMLU-Pro question (CoT is long), so always set `--limit` for first runs.

## Requirements

- Python 3.7+ (no external dependencies — uses only stdlib) for `test_lm_studio.py`, `test_context_window.py`, `test_decode_throughput.py`, `test_mmlu.py`
- [`uv`](https://docs.astral.sh/uv/) for `test_mmlu_pro.py` (auto-manages the `datasets` dependency)
- A running LLM server with OpenAI-compatible API (e.g., LM Studio)
