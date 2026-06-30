"""
Roofline Model and Performance Analysis for Attention Kernels.

Implements:
- Hardware specifications for common GPUs (A100, H100, T4, etc.)
- Arithmetic intensity calculation for attention at various configurations
- Achieved performance measurement
- Roofline plot generation
- Compute-bound vs memory-bound regime identification

Key insight: Standard attention is MEMORY-BOUND for typical sequence lengths
because it materializes an N×N attention matrix to HBM. FlashAttention makes
it more COMPUTE-BOUND by keeping intermediate results in SRAM.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HardwareSpec:
    """Hardware specifications for roofline analysis.

    All values are PEAK theoretical — achieved performance is always lower.
    """
    name: str
    # Peak compute in TFLOPS (fp16 tensor cores)
    peak_tflops_fp16: float
    # Peak HBM bandwidth in TB/s
    peak_bandwidth_tb_s: float
    # SRAM (shared memory) per SM in KB
    sram_per_sm_kb: float
    # Total SRAM across all SMs in KB
    total_sram_kb: float
    # Number of SMs
    num_sms: int
    # HBM capacity in GB
    hbm_gb: float

    @property
    def peak_flops_fp16(self) -> float:
        """Peak FLOPS in raw number (not TFLOPS)."""
        return self.peak_tflops_fp16 * 1e12

    @property
    def peak_bandwidth_bytes_s(self) -> float:
        """Peak bandwidth in bytes/second."""
        return self.peak_bandwidth_tb_s * 1e12

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at the ridge point (FLOPS/byte).

        Below this: memory-bound. Above this: compute-bound.
        """
        return self.peak_flops_fp16 / self.peak_bandwidth_bytes_s

    def achievable_flops(self, arithmetic_intensity: float) -> float:
        """Compute achievable FLOPS for a given arithmetic intensity.

        This is the roofline: min(peak_compute, peak_bandwidth * AI)
        """
        return min(
            self.peak_flops_fp16,
            self.peak_bandwidth_bytes_s * arithmetic_intensity,
        )


# Common GPU hardware specs
HARDWARE_SPECS = {
    "A100_80GB": HardwareSpec(
        name="NVIDIA A100 80GB",
        peak_tflops_fp16=312.0,
        peak_bandwidth_tb_s=2.039,
        sram_per_sm_kb=164,
        total_sram_kb=164 * 108,
        num_sms=108,
        hbm_gb=80,
    ),
    "A100_40GB": HardwareSpec(
        name="NVIDIA A100 40GB",
        peak_tflops_fp16=312.0,
        peak_bandwidth_tb_s=1.555,
        sram_per_sm_kb=164,
        total_sram_kb=164 * 108,
        num_sms=108,
        hbm_gb=40,
    ),
    "H100_SXM": HardwareSpec(
        name="NVIDIA H100 SXM",
        peak_tflops_fp16=989.0,
        peak_bandwidth_tb_s=3.35,
        sram_per_sm_kb=228,
        total_sram_kb=228 * 132,
        num_sms=132,
        hbm_gb=80,
    ),
    "T4": HardwareSpec(
        name="NVIDIA T4",
        peak_tflops_fp16=65.0,
        peak_bandwidth_tb_s=0.300,
        sram_per_sm_kb=64,
        total_sram_kb=64 * 40,
        num_sms=40,
        hbm_gb=16,
    ),
    "V100": HardwareSpec(
        name="NVIDIA V100",
        peak_tflops_fp16=125.0,
        peak_bandwidth_tb_s=0.900,
        sram_per_sm_kb=96,
        total_sram_kb=96 * 80,
        num_sms=80,
        hbm_gb=16,
    ),
    "RTX_4090": HardwareSpec(
        name="NVIDIA RTX 4090",
        peak_tflops_fp16=330.0,
        peak_bandwidth_tb_s=1.008,
        sram_per_sm_kb=100,
        total_sram_kb=100 * 128,
        num_sms=128,
        hbm_gb=24,
    ),
}


def detect_gpu_spec() -> Optional[HardwareSpec]:
    """Try to detect the current GPU and return its spec."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name().lower()
        if "a100" in name and "80" in name:
            return HARDWARE_SPECS["A100_80GB"]
        elif "a100" in name:
            return HARDWARE_SPECS["A100_40GB"]
        elif "h100" in name:
            return HARDWARE_SPECS["H100_SXM"]
        elif "t4" in name:
            return HARDWARE_SPECS["T4"]
        elif "v100" in name:
            return HARDWARE_SPECS["V100"]
        elif "4090" in name:
            return HARDWARE_SPECS["RTX_4090"]
        else:
            # Return A100 as default assumption
            return HARDWARE_SPECS["A100_80GB"]
    except Exception:
        return HARDWARE_SPECS["A100_80GB"]


@dataclass
class AttentionConfig:
    """Configuration for attention computation."""
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    causal: bool = False
    dtype_bytes: int = 2  # fp16 = 2 bytes


def compute_attention_flops(config: AttentionConfig) -> dict:
    """
    Compute total FLOPS for attention.

    Attention has two main matmuls:
    1. S = Q @ K^T: (N, d) × (d, N) → (N, N) — costs 2Nd per element, total 2N²d
    2. O = softmax(S) @ V: (N, N) × (N, d) → (N, d) — costs 2Nd per element, total 2N²d
    Plus softmax: ~5N² ops (exp, sum, div)

    Per head: ~4N²d + 5N²
    Total: batch × heads × (4N²d + 5N²)

    Args:
        config: Attention configuration

    Returns:
        Dict with FLOP breakdown
    """
    B = config.batch_size
    H = config.num_heads
    N = config.seq_len
    d = config.head_dim

    # QK^T matmul: 2 * N * N * d FLOPs per head
    qk_flops = 2 * N * N * d

    # Softmax: ~5N² (exp: N², sum: N², div: N², plus some extras)
    softmax_flops = 5 * N * N

    # Attn @ V matmul: 2 * N * N * d FLOPs per head
    av_flops = 2 * N * N * d

    flops_per_head = qk_flops + softmax_flops + av_flops
    total_flops = B * H * flops_per_head

    # For causal, roughly half the ops (lower triangle only)
    if config.causal:
        total_flops = total_flops // 2

    return {
        "qk_matmul_flops": B * H * qk_flops,
        "softmax_flops": B * H * softmax_flops,
        "attn_v_matmul_flops": B * H * av_flops,
        "total_flops": total_flops,
        "flops_per_head": flops_per_head,
    }


def compute_memory_traffic_naive(config: AttentionConfig) -> dict:
    """
    Compute HBM memory traffic for NAIVE (unfused) attention.

    The naive path materializes the full N×N attention matrix:
    1. Read Q, K → Compute S = QK^T → Write S to HBM
    2. Read S → Compute softmax(S) → Write P to HBM
    3. Read P, V → Compute O = PV → Write O to HBM

    Memory traffic: O(BHN²) — dominated by the attention matrix.

    Args:
        config: Attention configuration

    Returns:
        Dict with memory traffic breakdown in bytes
    """
    B = config.batch_size
    H = config.num_heads
    N = config.seq_len
    d = config.head_dim
    elem_size = config.dtype_bytes

    # Read Q and K for QK^T
    read_qk = 2 * B * H * N * d * elem_size

    # Write attention scores S (N×N matrix per head)
    write_s = B * H * N * N * elem_size

    # Read S for softmax, write P
    read_s = B * H * N * N * elem_size
    write_p = B * H * N * N * elem_size

    # Read P and V for PV matmul
    read_pv = B * H * N * N * elem_size + B * H * N * d * elem_size

    # Write output O
    write_o = B * H * N * d * elem_size

    total_bytes = read_qk + write_s + read_s + write_p + read_pv + write_o

    return {
        "read_qk": read_qk,
        "write_attention_matrix": write_s,
        "read_for_softmax": read_s,
        "write_softmax_result": write_p,
        "read_for_attn_v": read_pv,
        "write_output": write_o,
        "total_bytes": total_bytes,
        "attention_matrix_size_bytes": B * H * N * N * elem_size,
    }


def compute_memory_traffic_fused(config: AttentionConfig, sram_size_bytes: int = 192 * 1024) -> dict:
    """
    Compute HBM memory traffic for FUSED (FlashAttention) attention.

    The fused path never materializes the full N×N matrix. Instead:
    1. Read Q (once), stream K and V in blocks
    2. Accumulate output in SRAM
    3. Write final output

    Memory traffic: O(BHNd × N²d/M) where M is SRAM size
    In practice: O(BHN²d²/M + BHNd) — much less than O(BHN²) for typical d, M.

    The key insight: each element of K and V is read ceil(N/BLOCK_M) times
    (once per query block), not N times.

    Args:
        config: Attention configuration
        sram_size_bytes: Available SRAM per SM in bytes

    Returns:
        Dict with memory traffic breakdown in bytes
    """
    B = config.batch_size
    H = config.num_heads
    N = config.seq_len
    d = config.head_dim
    elem_size = config.dtype_bytes

    # Determine block sizes based on SRAM capacity
    # Need to fit: Q block (BLOCK_M × d) + K block (BLOCK_N × d) + V block (BLOCK_N × d)
    #            + O accumulator (BLOCK_M × d) + softmax stats (BLOCK_M)
    # Typical: BLOCK_M = BLOCK_N = 64 for d=64 on A100
    block_m = 64
    block_n = 64

    # Number of query blocks
    num_q_blocks = math.ceil(N / block_m)
    # Number of key/value blocks
    num_kv_blocks = math.ceil(N / block_n)

    # Q is read once (streamed through query blocks)
    read_q = B * H * N * d * elem_size

    # K and V are each read num_q_blocks times (for each query block)
    # This is the key difference vs FlashAttention-2 which reads K/V fewer times
    read_k = B * H * N * d * elem_size * num_q_blocks
    read_v = B * H * N * d * elem_size * num_q_blocks

    # Write output O (once)
    write_o = B * H * N * d * elem_size

    # Total — note this is typically MUCH less than naive when N >> d
    # because we avoid writing/reading the N×N attention matrix
    total_bytes = read_q + read_k + read_v + write_o

    # FlashAttention-2 optimization: swap loops so K/V are in outer loop
    # This reduces K/V reads to just once each:
    read_k_v2 = B * H * N * d * elem_size
    read_v_v2 = B * H * N * d * elem_size
    total_bytes_v2 = read_q + read_k_v2 + read_v_v2 + write_o

    return {
        "read_q": read_q,
        "read_k_v1": read_k,
        "read_v_v1": read_v,
        "read_k_v2": read_k_v2,
        "read_v_v2": read_v_v2,
        "write_output": write_o,
        "total_bytes_v1": total_bytes,
        "total_bytes_v2": total_bytes_v2,
        "num_q_blocks": num_q_blocks,
        "num_kv_blocks": num_kv_blocks,
        "block_m": block_m,
        "block_n": block_n,
    }


def compute_arithmetic_intensity(config: AttentionConfig, fused: bool = True) -> dict:
    """
    Compute arithmetic intensity (FLOPS / byte of HBM traffic).

    This determines whether the kernel is compute-bound or memory-bound
    on a given hardware target.

    Args:
        config: Attention configuration
        fused: Whether to use fused (FlashAttention) memory model

    Returns:
        Dict with arithmetic intensity and analysis
    """
    flops_info = compute_attention_flops(config)
    total_flops = flops_info["total_flops"]

    if fused:
        mem_info = compute_memory_traffic_fused(config)
        total_bytes = mem_info["total_bytes_v2"]  # Use V2 (optimized) estimate
    else:
        mem_info = compute_memory_traffic_naive(config)
        total_bytes = mem_info["total_bytes"]

    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0

    return {
        "arithmetic_intensity": arithmetic_intensity,  # FLOPS/byte
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "is_fused": fused,
    }


def roofline_analysis(
    config: AttentionConfig,
    hw: HardwareSpec,
    achieved_time_s: Optional[float] = None,
) -> dict:
    """
    Perform complete roofline analysis for an attention configuration.

    Args:
        config: Attention configuration
        hw: Hardware specification
        achieved_time_s: Measured kernel execution time (optional)

    Returns:
        Comprehensive analysis dict
    """
    # Compute for both fused and naive
    ai_fused = compute_arithmetic_intensity(config, fused=True)
    ai_naive = compute_arithmetic_intensity(config, fused=False)

    # Ridge point
    ridge = hw.ridge_point

    # Theoretical peak for each
    peak_fused = hw.achievable_flops(ai_fused["arithmetic_intensity"])
    peak_naive = hw.achievable_flops(ai_naive["arithmetic_intensity"])

    # Regime identification
    fused_regime = "compute-bound" if ai_fused["arithmetic_intensity"] > ridge else "memory-bound"
    naive_regime = "compute-bound" if ai_naive["arithmetic_intensity"] > ridge else "memory-bound"

    result = {
        "config": {
            "batch": config.batch_size,
            "heads": config.num_heads,
            "seq_len": config.seq_len,
            "head_dim": config.head_dim,
            "causal": config.causal,
        },
        "hardware": hw.name,
        "ridge_point": ridge,
        "fused": {
            "arithmetic_intensity": ai_fused["arithmetic_intensity"],
            "total_flops": ai_fused["total_flops"],
            "total_bytes": ai_fused["total_bytes"],
            "theoretical_peak_flops": peak_fused,
            "regime": fused_regime,
        },
        "naive": {
            "arithmetic_intensity": ai_naive["arithmetic_intensity"],
            "total_flops": ai_naive["total_flops"],
            "total_bytes": ai_naive["total_bytes"],
            "theoretical_peak_flops": peak_naive,
            "regime": naive_regime,
        },
    }

    # Add achieved performance if timing is provided
    if achieved_time_s is not None and achieved_time_s > 0:
        achieved_flops = ai_fused["total_flops"] / achieved_time_s
        efficiency = achieved_flops / hw.peak_flops_fp16 * 100
        result["achieved"] = {
            "time_s": achieved_time_s,
            "time_ms": achieved_time_s * 1000,
            "achieved_tflops": achieved_flops / 1e12,
            "hardware_utilization_pct": efficiency,
            "achieved_bandwidth_tb_s": ai_fused["total_bytes"] / achieved_time_s / 1e12,
        }

    return result


def sweep_sequence_lengths(
    hw: HardwareSpec,
    seq_lengths: List[int] = None,
    batch: int = 4,
    num_heads: int = 32,
    head_dim: int = 64,
    causal: bool = False,
) -> List[dict]:
    """
    Perform roofline analysis across multiple sequence lengths.

    This shows how arithmetic intensity changes with sequence length —
    the key insight of FlashAttention's advantage.

    Args:
        hw: Hardware specification
        seq_lengths: List of sequence lengths to analyze
        batch, num_heads, head_dim, causal: Attention parameters

    Returns:
        List of analysis results, one per sequence length
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    results = []
    for seq_len in seq_lengths:
        config = AttentionConfig(
            batch_size=batch,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            causal=causal,
        )
        analysis = roofline_analysis(config, hw)
        results.append(analysis)

    return results


if __name__ == "__main__":
    """Demo roofline analysis with A100 specs."""
    hw = HARDWARE_SPECS["A100_80GB"]
    print(f"Hardware: {hw.name}")
    print(f"  Peak FP16 compute: {hw.peak_tflops_fp16} TFLOPS")
    print(f"  Peak HBM bandwidth: {hw.peak_bandwidth_tb_s} TB/s")
    print(f"  Ridge point: {hw.ridge_point:.1f} FLOPS/byte")
    print(f"  SRAM per SM: {hw.sram_per_sm_kb} KB")
    print()

    print(f"{'Seq Len':>8} | {'AI (Fused)':>12} | {'AI (Naive)':>12} | {'Regime (F)':>14} | {'Regime (N)':>14}")
    print("-" * 75)

    results = sweep_sequence_lengths(hw, batch=4, num_heads=32, head_dim=64)
    for r in results:
        print(
            f"{r['config']['seq_len']:>8} | "
            f"{r['fused']['arithmetic_intensity']:>12.1f} | "
            f"{r['naive']['arithmetic_intensity']:>12.1f} | "
            f"{r['fused']['regime']:>14} | "
            f"{r['naive']['regime']:>14}"
        )
