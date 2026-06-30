"""
Benchmarking suite for attention kernel performance measurement.

Measures:
- Latency (ms) with proper GPU warm-up and synchronization
- Throughput (TFLOPS achieved)
- Memory usage (peak HBM allocation)
- Speedup vs baseline implementations

Supports benchmarking across multiple sequence lengths and head dimensions
to characterize performance regimes.
"""

import gc
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.roofline import AttentionConfig, compute_attention_flops


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    config: AttentionConfig
    # Timing
    latency_ms: float  # Median latency
    latency_std_ms: float  # Standard deviation
    latency_min_ms: float
    latency_max_ms: float
    # Performance
    achieved_tflops: float
    # Memory
    peak_memory_mb: float
    # Metadata
    num_warmup: int
    num_iterations: int


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results for comparison."""
    results: Dict[str, List[BenchmarkResult]] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        if result.name not in self.results:
            self.results[result.name] = []
        self.results[result.name].append(result)

    def get_speedups(self, target: str, baseline: str) -> List[float]:
        """Compute speedup of target over baseline at matching configs."""
        if target not in self.results or baseline not in self.results:
            return []
        speedups = []
        for t_result in self.results[target]:
            for b_result in self.results[baseline]:
                if (t_result.config.seq_len == b_result.config.seq_len and
                    t_result.config.head_dim == b_result.config.head_dim):
                    speedups.append(b_result.latency_ms / t_result.latency_ms)
        return speedups

    def to_table(self) -> List[dict]:
        """Convert results to a flat table format."""
        rows = []
        for name, results in self.results.items():
            for r in results:
                rows.append({
                    "kernel": name,
                    "seq_len": r.config.seq_len,
                    "head_dim": r.config.head_dim,
                    "batch": r.config.batch_size,
                    "heads": r.config.num_heads,
                    "causal": r.config.causal,
                    "latency_ms": r.latency_ms,
                    "tflops": r.achieved_tflops,
                    "memory_mb": r.peak_memory_mb,
                })
        return rows


def benchmark_attention_fn(
    fn: Callable,
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    causal: bool = False,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Tuple[float, float, float, float, float]:
    """
    Benchmark an attention function with proper GPU synchronization.

    Args:
        fn: Attention function with signature (q, k, v, causal) -> output
        q, k, v: Input tensors (already on GPU)
        causal: Whether to use causal masking
        num_warmup: Number of warmup iterations (not timed)
        num_iterations: Number of timed iterations

    Returns:
        Tuple of (median_ms, std_ms, min_ms, max_ms, peak_memory_mb)
    """
    assert HAS_TORCH and torch.cuda.is_available(), "CUDA required for benchmarking"

    device = q.device

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmup):
        _ = fn(q, k, v, causal=causal)
    torch.cuda.synchronize()

    # Timed iterations using CUDA events for precise timing
    timings = []
    for _ in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = fn(q, k, v, causal=causal)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        timings.append(elapsed_ms)

    timings = np.array(timings)
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return (
        float(np.median(timings)),
        float(np.std(timings)),
        float(np.min(timings)),
        float(np.max(timings)),
        peak_memory_mb,
    )


def run_benchmark(
    name: str,
    fn: Callable,
    config: AttentionConfig,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> BenchmarkResult:
    """
    Run a complete benchmark for one attention function at one configuration.

    Args:
        name: Name of the kernel being benchmarked
        fn: Attention function (q, k, v, causal) -> output
        config: Attention configuration
        num_warmup: Warmup iterations
        num_iterations: Timed iterations

    Returns:
        BenchmarkResult with timing and performance data
    """
    assert HAS_TORCH and torch.cuda.is_available()

    from src.reference import generate_test_inputs

    # Generate inputs
    q, k, v = generate_test_inputs(
        batch=config.batch_size,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        head_dim=config.head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    # Run benchmark
    median_ms, std_ms, min_ms, max_ms, peak_mem = benchmark_attention_fn(
        fn, q, k, v, causal=config.causal,
        num_warmup=num_warmup, num_iterations=num_iterations,
    )

    # Compute achieved TFLOPS
    flops_info = compute_attention_flops(config)
    total_flops = flops_info["total_flops"]
    achieved_tflops = total_flops / (median_ms / 1000) / 1e12 if median_ms > 0 else 0

    # Clean up
    del q, k, v
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult(
        name=name,
        config=config,
        latency_ms=median_ms,
        latency_std_ms=std_ms,
        latency_min_ms=min_ms,
        latency_max_ms=max_ms,
        achieved_tflops=achieved_tflops,
        peak_memory_mb=peak_mem,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )


def run_benchmark_sweep(
    kernels: Dict[str, Callable],
    seq_lengths: List[int] = None,
    head_dims: List[int] = None,
    batch: int = 4,
    num_heads: int = 32,
    causal: bool = False,
    num_warmup: int = 10,
    num_iterations: int = 50,
) -> BenchmarkSuite:
    """
    Run benchmarks across multiple configurations.

    Args:
        kernels: Dict mapping kernel name to function
        seq_lengths: Sequence lengths to benchmark
        head_dims: Head dimensions to benchmark
        batch: Batch size
        num_heads: Number of attention heads
        causal: Whether to use causal masking
        num_warmup: Warmup iterations per config
        num_iterations: Timed iterations per config

    Returns:
        BenchmarkSuite with all results
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    if head_dims is None:
        head_dims = [64]

    suite = BenchmarkSuite()

    total_configs = len(kernels) * len(seq_lengths) * len(head_dims)
    print(f"Running {total_configs} benchmark configurations...")
    print(f"  Kernels: {list(kernels.keys())}")
    print(f"  Seq lengths: {seq_lengths}")
    print(f"  Head dims: {head_dims}")
    print(f"  Batch={batch}, Heads={num_heads}, Causal={causal}")
    print()

    count = 0
    for head_dim in head_dims:
        for seq_len in seq_lengths:
            config = AttentionConfig(
                batch_size=batch,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                causal=causal,
            )

            for name, fn in kernels.items():
                count += 1
                try:
                    result = run_benchmark(
                        name=name,
                        fn=fn,
                        config=config,
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                    )
                    suite.add_result(result)
                    print(
                        f"  [{count}/{total_configs}] {name:20s} | "
                        f"seq={seq_len:5d} d={head_dim:3d} | "
                        f"{result.latency_ms:8.3f} ms | "
                        f"{result.achieved_tflops:6.2f} TFLOPS | "
                        f"{result.peak_memory_mb:8.1f} MB"
                    )
                except Exception as e:
                    print(f"  [{count}/{total_configs}] {name:20s} | seq={seq_len:5d} | FAILED: {e}")

    return suite


def estimate_memory_savings(config: AttentionConfig) -> dict:
    """
    Estimate memory savings from fused vs naive attention.

    The naive approach materializes the N×N attention matrix.
    The fused approach only needs O(N) extra memory.

    Args:
        config: Attention configuration

    Returns:
        Dict with memory analysis
    """
    B = config.batch_size
    H = config.num_heads
    N = config.seq_len
    d = config.head_dim
    elem_size = config.dtype_bytes

    # Input memory (same for both)
    input_mem = 3 * B * H * N * d * elem_size  # Q, K, V
    output_mem = B * H * N * d * elem_size  # O

    # Naive: materializes full attention matrix
    attn_matrix_mem = B * H * N * N * elem_size
    naive_total = input_mem + output_mem + attn_matrix_mem

    # Fused: only needs block accumulators + LSE buffer
    # Accumulators: BLOCK_M × d (in SRAM, not HBM)
    # LSE buffer: B × H × N × 4 bytes (fp32)
    lse_mem = B * H * N * 4
    fused_total = input_mem + output_mem + lse_mem

    savings_bytes = naive_total - fused_total
    savings_pct = savings_bytes / naive_total * 100

    return {
        "naive_total_mb": naive_total / (1024 * 1024),
        "fused_total_mb": fused_total / (1024 * 1024),
        "savings_mb": savings_bytes / (1024 * 1024),
        "savings_pct": savings_pct,
        "attention_matrix_mb": attn_matrix_mem / (1024 * 1024),
        "seq_len": N,
    }


if __name__ == "__main__":
    """Demo benchmark analysis (theoretical, no GPU needed)."""
    print("Memory Savings Analysis (Fused vs Naive)")
    print("=" * 60)
    print(f"{'Seq Len':>8} | {'Naive (MB)':>11} | {'Fused (MB)':>11} | {'Savings':>9} | {'Attn Matrix (MB)':>17}")
    print("-" * 60)

    for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        config = AttentionConfig(
            batch_size=4, num_heads=32, seq_len=seq_len, head_dim=64
        )
        mem = estimate_memory_savings(config)
        print(
            f"{seq_len:>8} | "
            f"{mem['naive_total_mb']:>11.1f} | "
            f"{mem['fused_total_mb']:>11.1f} | "
            f"{mem['savings_pct']:>7.1f}% | "
            f"{mem['attention_matrix_mb']:>17.1f}"
        )
