# Benchmarks

All benchmarks run using LM Studio as the inference server.

## Test Machines

| Machine | CPU/GPU | RAM/VRAM |
|---------|---------|----------|
| MacBook Pro | M1 Max | 64GB unified |
| Desktop | RTX 4070 Ti Super | 16GB VRAM |

## Models Tested

| Model | Type | Quantization | Parameters |
|-------|------|-------------|------------|
| qwen/qwen3.5-35b-a3b | MoE | GGUF Q4_K_M | 35B (3B active) |
| qwen3.5-9b | Dense | - | 9B |
| lmstudio-community/qwen3.5-27b | Dense | GGUF Q4_K_M | 27B |
| mlx-community/qwen3.5-27b | Dense | MLX 4-bit | 27B |
| unsloth/qwen3.5-27b | Dense | GGUF | 27B |

---

## Prefill (Prompt Processing) Benchmarks

### qwen/qwen3.5-35b-a3b (MoE, GGUF Q4_K_M) — M1 Max

Full 256K context supported with recall.

| Size | Time | Prompt tok/s | Completion tok/s | Recall |
|------|------|--------------|------------------|--------|
| 1K | 6s | 131.0 | 16.9 | Pass |
| 64K | 3.3 min | 284.5 | 5.5 | Pass |
| 128K | 7 min | 264.2 | 0.2 | Pass |
| 192K | 14 min | 208.3 | 0.1 | Pass |
| 256K | 23 min | 166.4 | 0.1 | Pass |

### qwen3.5-9b — M1 Max

| Size | Time | Prompt tok/s | Completion tok/s | Recall |
|------|------|--------------|------------------|--------|
| 1K | 55s | 14.0 | 6.4 | Pass |
| 64K | 2 min 23s | 397.6 | 2.6 | Pass |

### qwen3.5-9b — 4070 Ti Super

| Size | Time | Prompt tok/s | Completion tok/s | Recall |
|------|------|--------------|------------------|--------|
| 1K | 0.7s | 1,059 | 12.3 | Pass |
| 64K | 19s | **2,973** | 0.5 | Pass |

### lmstudio-community/qwen3.5-27b (Dense, GGUF Q4_K_M) — M1 Max

| Size | Time | Prompt tok/s | Completion tok/s | Recall |
|------|------|--------------|------------------|--------|
| 1K | 51s | 15.1 | 6.9 | Pass |
| 16K | 195s | 72.4 | 2.2 | Pass |

### mlx-community/qwen3.5-27b (Dense, MLX 4-bit) — M1 Max

| Size | Time | Prompt tok/s | Completion tok/s | Recall |
|------|------|--------------|------------------|--------|
| 1K | 33s | 23.5 | 11.3 | Pass |
| 16K | 210s | 67.1 | 3.3 | Pass |
| 64K | 13 min | 72.0 | 0.4 | Pass |
| 128K | 31 min | 61.1 | 0.3 | Pass |
| 192K | 55 min | 51.7 | 0.1 | Pass |

### unsloth/qwen3.5-27b (Dense, GGUF) — M1 Max

| Size | Time | Prompt tok/s | Completion tok/s | Recall |
|------|------|--------------|------------------|--------|
| 1K | 21s | 37.3 | 4.8 | Pass |
| 128K | 27 min | 69.6 | 0.1 | Fail* |
| 192K | >30 min | - | - | Timeout |

*128K recall failure was due to max_tokens=100 (too low to capture thinking + answer). Later tests used max_tokens=4096.

### Cold vs Warm — qwen3.5-27b on M1 Max

| Size | Time | Prompt tok/s | Completion tok/s | Recall |
|------|------|--------------|------------------|--------|
| 64K (cold) | 11 min 43s | 80.7 | 0.5 | Pass |
| 64K (warm) | 2 min 23s | 397.6 | 2.6 | Pass |

---

## Decode (Generation) Benchmarks

### qwen3.5-9b — 4070 Ti Super

| Target | Actual | Time | Decode tok/s |
|--------|--------|------|-------------|
| 256 | 256 | 4.3s | 59.2 |
| 32K | 32,768 | 8 min 24s | **65.1** |

### qwen3.5-9b — M1 Max

| Target | Actual | Time | Decode tok/s |
|--------|--------|------|-------------|
| 256 | 256 | 12s | 22.0 |
| 8K | 8,192 | 5 min 46s | 23.7 |

---

## Key Findings

### MoE vs Dense — Prefill at 64K

| Model | Type | Prompt tok/s | Time |
|-------|------|--------------|------|
| qwen3.5-35b-a3b | MoE | **284.5** | 3.3 min |
| mlx-community/qwen3.5-27b | Dense | 72.0 | 13 min |

The MoE model is ~4x faster on prompt processing because it activates only ~3B parameters per token vs all 27B for the dense model.

### GGUF vs MLX — Prefill at 16K (Dense 27B)

| Model | Format | Prompt tok/s | Completion tok/s |
|-------|--------|--------------|------------------|
| lmstudio-community/qwen3.5-27b | GGUF Q4_K_M | **72.4** | 2.2 |
| mlx-community/qwen3.5-27b | MLX 4-bit | 67.1 | **3.3** |

GGUF (llama.cpp) is ~8% faster on prompt processing. MLX is slightly faster on completion.

### 4070 Ti Super vs M1 Max — qwen3.5-9b

| Metric | 4070 Ti Super | M1 Max |
|--------|--------------|--------|
| Prefill (64K) | **2,973 tok/s** | 398 tok/s |
| Decode | **65.1 tok/s** | 23.7 tok/s |

The 4070 Ti Super is ~7.5x faster on prefill and ~2.7x faster on decode for this 9B model.

### Throughput vs Context Size

Prompt throughput generally decreases as context size increases due to quadratic attention costs (qwen3.5-35b-a3b MoE):
- 64K: 284 tok/s
- 128K: 264 tok/s
- 192K: 208 tok/s
- 256K: 166 tok/s

### Cold vs Warm Model

After reloading the model, 64K prompt throughput dropped from ~398 tok/s to ~81 tok/s, reflecting the cost of loading the model into memory and initial GPU warm-up.
