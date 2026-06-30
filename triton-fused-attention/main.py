#!/usr/bin/env python3
"""
Triton Fused Attention — FlashAttention Kernel + Roofline Analysis
===================================================================

Main entry point that runs:
1. Correctness validation (fused kernel vs reference)
2. Roofline performance analysis (theoretical + measured)
3. Benchmarking across sequence lengths
4. Visualization and report generation

Usage:
    # Full analysis with GPU benchmarks
    python main.py

    # Theoretical analysis only (no GPU needed)
    python main.py --theoretical-only

    # Quick run with fewer configurations
    python main.py --quick
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.roofline import (
    HARDWARE_SPECS,
    HardwareSpec,
    AttentionConfig,
    compute_attention_flops,
    compute_arithmetic_intensity,
    compute_memory_traffic_naive,
    compute_memory_traffic_fused,
    roofline_analysis,
    sweep_sequence_lengths,
    detect_gpu_spec,
)
from src.benchmark import (
    BenchmarkSuite,
    run_benchmark_sweep,
    estimate_memory_savings,
)
from src.visualization import (
    plot_roofline,
    plot_speedup_curve,
    plot_memory_savings,
    plot_arithmetic_intensity_analysis,
    generate_all_plots,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triton Fused Attention — FlashAttention Kernel + Roofline Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--theoretical-only", action="store_true",
        help="Run only theoretical analysis (no GPU required)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with fewer benchmark configurations"
    )
    parser.add_argument(
        "--hardware", type=str, default="auto",
        choices=["auto", "A100_80GB", "A100_40GB", "H100_SXM", "T4", "V100", "RTX_4090"],
        help="Hardware spec to use for roofline analysis"
    )
    parser.add_argument(
        "--batch", type=int, default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=32,
        help="Number of attention heads (default: 32)"
    )
    parser.add_argument(
        "--head-dim", type=int, default=64,
        help="Head dimension (default: 64)"
    )
    parser.add_argument(
        "--causal", action="store_true",
        help="Use causal (autoregressive) masking"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save results"
    )
    return parser.parse_args()


def print_header():
    print("=" * 72)
    print("  Triton Fused Attention — FlashAttention Kernel + Roofline Analysis")
    print("=" * 72)


def run_theoretical_analysis(hw: HardwareSpec, args) -> dict:
    """Run purely theoretical analysis (no GPU needed)."""
    print(f"\n{'THEORETICAL ANALYSIS':=^72}")
    print(f"\nHardware target: {hw.name}")
    print(f"  Peak FP16 compute:    {hw.peak_tflops_fp16:.0f} TFLOPS")
    print(f"  Peak HBM bandwidth:   {hw.peak_bandwidth_tb_s:.3f} TB/s")
    print(f"  Ridge point:          {hw.ridge_point:.0f} FLOPS/byte")
    print(f"  SRAM per SM:          {hw.sram_per_sm_kb:.0f} KB")

    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # ── FLOPS Analysis ──
    print(f"\n{'─── FLOPS Analysis ───':─^72}")
    print(f"  Config: batch={args.batch}, heads={args.num_heads}, d={args.head_dim}")
    print(f"\n  {'Seq Len':>8} | {'Total GFLOPS':>13} | {'QK^T GFLOPS':>12} | {'AV GFLOPS':>10}")
    print(f"  {'-'*52}")

    for N in seq_lengths:
        config = AttentionConfig(args.batch, args.num_heads, N, args.head_dim, args.causal)
        flops = compute_attention_flops(config)
        total_g = flops['total_flops'] / 1e9
        qk_g = flops['qk_matmul_flops'] / 1e9
        av_g = flops['attn_v_matmul_flops'] / 1e9
        print(f"  {N:>8} | {total_g:>13.1f} | {qk_g:>12.1f} | {av_g:>10.1f}")

    # ── Memory Traffic Analysis ──
    print(f"\n{'─── Memory Traffic Analysis ───':─^72}")
    print(f"\n  {'Seq Len':>8} | {'Naive (MB)':>10} | {'Fused (MB)':>10} | {'Reduction':>10} | {'Attn Matrix':>11}")
    print(f"  {'-'*60}")

    for N in seq_lengths:
        config = AttentionConfig(args.batch, args.num_heads, N, args.head_dim, args.causal)
        naive_mem = compute_memory_traffic_naive(config)
        fused_mem = compute_memory_traffic_fused(config)
        naive_mb = naive_mem['total_bytes'] / 1e6
        fused_mb = fused_mem['total_bytes_v2'] / 1e6
        reduction = naive_mb / fused_mb if fused_mb > 0 else 0
        attn_matrix_mb = naive_mem['attention_matrix_size_bytes'] / 1e6
        print(f"  {N:>8} | {naive_mb:>10.1f} | {fused_mb:>10.1f} | {reduction:>9.1f}× | {attn_matrix_mb:>11.1f}")

    # ── Arithmetic Intensity & Regime ──
    print(f"\n{'─── Arithmetic Intensity & Regime ───':─^72}")
    print(f"  Ridge point: {hw.ridge_point:.0f} FLOPS/byte")
    print(f"  (Below ridge = memory-bound, Above = compute-bound)")
    print(f"\n  {'Seq Len':>8} | {'AI (Fused)':>11} | {'AI (Naive)':>11} | {'Fused Regime':>14} | {'Naive Regime':>13}")
    print(f"  {'-'*66}")

    analysis_results = []
    for N in seq_lengths:
        config = AttentionConfig(args.batch, args.num_heads, N, args.head_dim, args.causal)
        result = roofline_analysis(config, hw)
        analysis_results.append(result)
        print(
            f"  {N:>8} | "
            f"{result['fused']['arithmetic_intensity']:>11.1f} | "
            f"{result['naive']['arithmetic_intensity']:>11.1f} | "
            f"{result['fused']['regime']:>14} | "
            f"{result['naive']['regime']:>13}"
        )

    # ── Memory Savings ──
    print(f"\n{'─── HBM Memory Savings (Activation Memory) ───':─^72}")
    print(f"\n  {'Seq Len':>8} | {'Naive (MB)':>10} | {'Fused (MB)':>10} | {'Savings':>8} | {'N×N Matrix (MB)':>15}")
    print(f"  {'-'*60}")

    for N in seq_lengths:
        config = AttentionConfig(args.batch, args.num_heads, N, args.head_dim, args.causal)
        mem = estimate_memory_savings(config)
        print(
            f"  {N:>8} | "
            f"{mem['naive_total_mb']:>10.1f} | "
            f"{mem['fused_total_mb']:>10.1f} | "
            f"{mem['savings_pct']:>7.1f}% | "
            f"{mem['attention_matrix_mb']:>15.1f}"
        )

    # ── Key Insight ──
    print(f"\n{'─── KEY INSIGHT ───':─^72}")
    print(f"""
  Standard attention is MEMORY-BOUND because it materializes an O(N²) attention
  matrix to HBM. At seq_len=4096, the attention matrix alone is 
  {args.batch * args.num_heads * 4096 * 4096 * 2 / 1e9:.1f} GB.

  FlashAttention keeps the attention scores in SRAM ({hw.sram_per_sm_kb:.0f} KB/SM),
  never writing the full N×N matrix to HBM. This reduces memory traffic by
  O(N/BLOCK_SIZE)× and shifts the kernel from memory-bound to compute-bound.

  On {hw.name}:
  - Naive at N=4096: AI = {analysis_results[5]['naive']['arithmetic_intensity']:.1f} FLOPS/byte → {analysis_results[5]['naive']['regime']}
  - Fused at N=4096: AI = {analysis_results[5]['fused']['arithmetic_intensity']:.1f} FLOPS/byte → {analysis_results[5]['fused']['regime']}
""")

    return {"analysis_results": analysis_results, "hardware": hw.name}


def run_gpu_benchmarks(hw: HardwareSpec, args) -> BenchmarkSuite:
    """Run GPU benchmarks (requires CUDA)."""
    import torch
    from src.kernel import flash_attention_forward
    from src.reference import naive_attention, pytorch_sdpa_attention

    print(f"\n{'GPU BENCHMARKS':=^72}")
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")

    # Define kernel functions with consistent interface
    def triton_flash(q, k, v, causal=False):
        return flash_attention_forward(q, k, v, causal=causal)

    def naive(q, k, v, causal=False):
        return naive_attention(q, k, v, causal=causal)

    def pytorch_sdpa(q, k, v, causal=False):
        return pytorch_sdpa_attention(q, k, v, causal=causal)

    kernels = {
        "triton_flash": triton_flash,
        "naive": naive,
        "pytorch_sdpa": pytorch_sdpa,
    }

    if args.quick:
        seq_lengths = [256, 512, 1024, 2048]
        num_iterations = 20
    else:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        num_iterations = 50

    suite = run_benchmark_sweep(
        kernels=kernels,
        seq_lengths=seq_lengths,
        head_dims=[args.head_dim],
        batch=args.batch,
        num_heads=args.num_heads,
        causal=args.causal,
        num_warmup=10,
        num_iterations=num_iterations,
    )

    # Print summary
    print(f"\n{'─── Speedup Summary ───':─^72}")
    if 'triton_flash' in suite.results and 'naive' in suite.results:
        print(f"\n  {'Seq Len':>8} | {'Triton (ms)':>12} | {'Naive (ms)':>10} | {'PyTorch (ms)':>12} | {'Speedup vs Naive':>16}")
        print(f"  {'-'*68}")
        for t_res in suite.results['triton_flash']:
            naive_res = None
            sdpa_res = None
            for n in suite.results.get('naive', []):
                if n.config.seq_len == t_res.config.seq_len:
                    naive_res = n
                    break
            for s in suite.results.get('pytorch_sdpa', []):
                if s.config.seq_len == t_res.config.seq_len:
                    sdpa_res = s
                    break
            naive_ms = naive_res.latency_ms if naive_res else float('nan')
            sdpa_ms = sdpa_res.latency_ms if sdpa_res else float('nan')
            speedup = naive_ms / t_res.latency_ms if naive_res else float('nan')
            print(
                f"  {t_res.config.seq_len:>8} | "
                f"{t_res.latency_ms:>12.3f} | "
                f"{naive_ms:>10.3f} | "
                f"{sdpa_ms:>12.3f} | "
                f"{speedup:>16.2f}×"
            )

    return suite


def run_correctness_check(args):
    """Validate the Triton kernel produces correct results."""
    import torch
    from src.kernel import flash_attention_forward
    from src.reference import validate_correctness, generate_test_inputs

    print(f"\n{'CORRECTNESS VALIDATION':=^72}")

    all_pass = True
    configs = [
        (1, 4, 128, 64),
        (2, 8, 256, 64),
        (2, 8, 512, 64),
        (4, 16, 1024, 64),
        (2, 8, 512, 128),
    ]

    for batch, heads, seq_len, head_dim in configs:
        for causal in [False, True]:
            q, k, v = generate_test_inputs(
                batch=batch, num_heads=heads, seq_len=seq_len,
                head_dim=head_dim, device="cuda"
            )

            def test_fn(q, k, v, causal=causal):
                return flash_attention_forward(q, k, v, causal=causal)

            results = validate_correctness(test_fn, q, k, v, causal=causal)
            status = "✓ PASS" if results["is_correct"] else "✗ FAIL"
            if not results["is_correct"]:
                all_pass = False

            print(
                f"  {status} | B={batch} H={heads:2d} N={seq_len:5d} d={head_dim:3d} "
                f"causal={str(causal):5s} | "
                f"max_err={results['max_abs_error']:.6f} "
                f"cos_sim={results['cosine_similarity']:.8f}"
            )

            del q, k, v
            torch.cuda.empty_cache()

    print(f"\n  {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    return all_pass


def main():
    args = parse_args()
    print_header()

    # Determine hardware
    if args.hardware == "auto":
        hw = detect_gpu_spec()
        if hw is None:
            hw = HARDWARE_SPECS["A100_80GB"]
    else:
        hw = HARDWARE_SPECS[args.hardware]

    results_dir = Path(PROJECT_ROOT) / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Configuration:")
    print(f"    Hardware:    {hw.name}")
    print(f"    Batch:       {args.batch}")
    print(f"    Heads:       {args.num_heads}")
    print(f"    Head dim:    {args.head_dim}")
    print(f"    Causal:      {args.causal}")
    print(f"    Results dir: {results_dir}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Theoretical Analysis (always runs)
    # ═══════════════════════════════════════════════════════════════════
    theory_results = run_theoretical_analysis(hw, args)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: GPU Benchmarks (if available and not theoretical-only)
    # ═══════════════════════════════════════════════════════════════════
    suite = None
    measured_points = None

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if has_cuda and not args.theoretical_only:
        # Correctness check first
        all_correct = run_correctness_check(args)
        if not all_correct:
            print("\n  WARNING: Correctness check failed. Benchmark results may be invalid.")

        # Run benchmarks
        suite = run_gpu_benchmarks(hw, args)

        # Collect measured points for roofline
        measured_points = []
        if 'triton_flash' in suite.results:
            for r in suite.results['triton_flash']:
                config = r.config
                ai = compute_arithmetic_intensity(config, fused=True)
                measured_points.append({
                    'arithmetic_intensity': ai['arithmetic_intensity'],
                    'achieved_tflops': r.achieved_tflops,
                    'label': f'N={config.seq_len}',
                })
    elif not has_cuda:
        print(f"\n  NOTE: No CUDA GPU detected. Running theoretical analysis only.")
        print(f"  To run GPU benchmarks, use a machine with an NVIDIA GPU.")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Generate Visualizations
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'GENERATING VISUALIZATIONS':=^72}")
    generate_all_plots(hw, suite=suite, measured_points=measured_points,
                       save_dir=str(results_dir))

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Save Results
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'SAVING RESULTS':=^72}")

    # Save analysis summary
    summary = {
        "hardware": hw.name,
        "config": {
            "batch": args.batch,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "causal": args.causal,
        },
        "kernel_info": {
            "name": "TritonFlashAttention",
            "algorithm": "FlashAttention (online softmax, tiled)",
            "memory_complexity": "O(N) — no materialized attention matrix",
            "compute_complexity": "O(N²d)",
        },
        "ridge_point_flops_per_byte": hw.ridge_point,
    }

    if suite and 'triton_flash' in suite.results:
        summary["benchmark_results"] = suite.to_table()

    summary_path = results_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    # ═══════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'SUMMARY':=^72}")
    print(f"""
  Kernel: FlashAttention-style fused attention in Triton
  Hardware: {hw.name} (peak: {hw.peak_tflops_fp16} TFLOPS, {hw.peak_bandwidth_tb_s} TB/s)

  Key Results:
  • Fused kernel eliminates O(N²) HBM traffic from attention matrix
  • Arithmetic intensity increases from ~{theory_results['analysis_results'][3]['naive']['arithmetic_intensity']:.0f} (naive) 
    to ~{theory_results['analysis_results'][3]['fused']['arithmetic_intensity']:.0f} FLOPS/byte (fused) at N=1024
  • Regime shift: naive is memory-bound, fused becomes compute-bound
  • Memory savings: >90% at seq_len ≥ 2048
""")

    if suite and 'triton_flash' in suite.results and 'naive' in suite.results:
        # Find max speedup
        max_speedup = 0
        max_seq = 0
        for t_res in suite.results['triton_flash']:
            for n_res in suite.results['naive']:
                if t_res.config.seq_len == n_res.config.seq_len:
                    s = n_res.latency_ms / t_res.latency_ms
                    if s > max_speedup:
                        max_speedup = s
                        max_seq = t_res.config.seq_len
        print(f"  • Peak measured speedup: {max_speedup:.2f}× at seq_len={max_seq}")

    print(f"\n  Results saved to: {results_dir}/")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
