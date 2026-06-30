# Triton Fused Attention — FlashAttention Kernel + Roofline Analysis

A custom fused attention kernel written in [Triton](https://github.com/openai/triton), implementing the FlashAttention algorithm with a complete roofline performance model to characterize compute vs memory boundedness.

## What This Demonstrates

1. **Kernel Engineering** — Writing a GPU kernel that tiles Q/K/V matrices into SRAM-sized blocks, fuses softmax + matmul into a single pass, and uses online softmax for numerical stability
2. **Performance Characterization** — Building a roofline model to identify whether the kernel is compute-bound or memory-bound, and how that changes with sequence length
3. **Hardware Understanding** — Reasoning about memory hierarchy (HBM vs SRAM), arithmetic intensity, and how kernel fusion changes the performance regime

## The Core Insight

Standard attention materializes a full N×N attention matrix to HBM:

```
Q @ K^T → [write N×N to HBM] → softmax → [write N×N to HBM] → @ V → O
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           This HBM traffic makes standard attention MEMORY-BOUND
```

FlashAttention keeps everything in SRAM:

```
for each Q_block:                     ← Tile queries into SRAM blocks
    for each K_block, V_block:        ← Stream K/V from HBM in blocks
        S_block = Q_block @ K_block^T ← In SRAM: never touches HBM
        P_block = online_softmax(S)   ← In SRAM: numerically stable
        O_block += P_block @ V_block  ← In SRAM: accumulate output
    write O_block to HBM              ← Only final output touches HBM
```

This reduces memory traffic from O(N²) to O(N), shifting the kernel from **memory-bound** to **compute-bound**.

## Project Structure

```
triton-fused-attention/
├── main.py                  # Main entry point — runs full analysis
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── kernel.py            # Triton fused attention kernel (the core)
│   ├── reference.py         # Naive + PyTorch reference implementations
│   ├── roofline.py          # Roofline model + hardware specs
│   ├── benchmark.py         # GPU benchmarking suite
│   └── visualization.py     # Plotting utilities
├── results/                 # Generated plots and analysis
└── scripts/                 # Utility scripts
```

## Quick Start

### Prerequisites

- NVIDIA GPU (Volta or newer — V100, T4, A100, H100, RTX 30xx/40xx)
- CUDA toolkit installed
- Python 3.8+

### Install

```bash
pip install torch triton numpy matplotlib seaborn pandas tabulate
```

### Run

```bash
# Full analysis (theoretical + GPU benchmarks + plots)
python main.py

# Theoretical analysis only (no GPU needed — still generates roofline plots)
python main.py --theoretical-only

# Quick benchmark (fewer configs)
python main.py --quick

# With causal masking
python main.py --causal

# Specify hardware for roofline (if auto-detect fails)
python main.py --hardware A100_80GB
```

## The Kernel: How It Works

### Algorithm: Online Softmax + Tiled Attention

The key challenge in fusing attention is that softmax requires knowing the max of the entire row before computing exponentials. The **online softmax** trick solves this:

```python
# For each query block:
m_i = -inf     # Running max
l_i = 0        # Running sum of exp
o_i = 0        # Running output

for each K_block, V_block:
    s = Q_block @ K_block^T * scale    # Local attention scores
    m_new = max(m_i, max(s))           # Update running max
    
    # Correction: rescale previous accumulator
    alpha = exp(m_i - m_new)
    l_new = alpha * l_i + sum(exp(s - m_new))
    
    # Accumulate output with new scaling
    o_i = alpha * o_i + exp(s - m_new) @ V_block
    
    m_i = m_new
    l_i = l_new

output = o_i / l_i  # Final normalization
```

### Triton Implementation Details

- **Block sizes**: BLOCK_M=64, BLOCK_N=64 (tunable per hardware)
- **Memory layout**: (batch×heads, seq_len, head_dim) for coalesced access
- **Precision**: fp16 inputs, fp32 accumulation for stability
- **Causal masking**: Early termination of inner loop (fewer K/V blocks processed)

## Roofline Analysis

### Reading the Roofline Plot

```
Performance (TFLOPS)
    │         ┌──────── Compute ceiling (peak TFLOPS)
    │         │
    │    ─────┼──────── 
    │   /     │        
    │  / ←Ridge       Compute-bound region
    │ /  point        (limited by ALU throughput)
    │/                 
    │  Memory-bound
    │  region (limited
    │  by HBM bandwidth)
    └──────────────────── Arithmetic Intensity (FLOPS/byte)
```

### Key Numbers (A100 80GB)

| Metric | Value |
|--------|-------|
| Peak FP16 | 312 TFLOPS |
| HBM Bandwidth | 2.0 TB/s |
| Ridge Point | 153 FLOPS/byte |
| SRAM per SM | 164 KB |

### Regime Analysis

| Seq Length | Naive AI | Fused AI | Naive Regime | Fused Regime |
|-----------|----------|----------|--------------|--------------|
| 512 | ~32 | ~128 | Memory-bound | Memory-bound |
| 1024 | ~32 | ~256 | Memory-bound | **Compute-bound** |
| 4096 | ~32 | ~1024 | Memory-bound | **Compute-bound** |

The naive approach stays memory-bound regardless of sequence length (AI ≈ d/4 ≈ 32 for d=128). The fused approach's AI grows with sequence length because it amortizes HBM reads over more computation.

## Memory Savings

At seq_len=4096 with batch=4, heads=32, d=64:

| | Naive | Fused | Savings |
|---|---|---|---|
| **Attention matrix** | 4 GB | 0 | 100% |
| **Total activation memory** | 4.1 GB | 0.1 GB | **97%** |

This is why FlashAttention enables training with longer sequences — the O(N²) memory barrier is removed.

## Performance Comparison

Expected results on A100 (batch=4, heads=32, d=64):

| Seq Length | Triton Flash | Naive | Speedup |
|-----------|-------------|-------|---------|
| 512 | ~0.5ms | ~0.8ms | ~1.6× |
| 1024 | ~1.2ms | ~2.5ms | ~2.1× |
| 2048 | ~3.5ms | ~9.0ms | ~2.6× |
| 4096 | ~12ms | ~35ms | ~2.9× |

Speedup increases with sequence length because the memory traffic reduction becomes more significant.


## Extending This Project

- **Backward pass**: Implement the FlashAttention backward kernel (recomputes attention during backward to avoid storing the N×N matrix)
- **Multi-query attention / GQA**: Adapt for shared K/V heads
- **Variable-length batching**: Handle sequences of different lengths without padding
- **Autotuning**: Use Triton's `@triton.autotune` to find optimal block sizes per hardware

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [Roofline: An Insightful Visual Performance Model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyworksOriginal.pdf) (Williams et al., 2009)

## License

MIT
