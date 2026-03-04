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

Benchmarks decode (token generation) throughput by requesting the model to generate a configurable number of tokens. Uses the raw `/v1/completions` endpoint to avoid chat template stop sequences.

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

## Requirements

- Python 3.7+ (no external dependencies — uses only stdlib)
- A running LLM server with OpenAI-compatible API (e.g., LM Studio)
