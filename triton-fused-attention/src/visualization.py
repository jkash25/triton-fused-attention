"""
Visualization module for roofline analysis and benchmark results.

Generates:
- Roofline plots (log-log: arithmetic intensity vs FLOPS)
- Speedup curves across sequence lengths
- Memory savings comparison
- Performance breakdown charts
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from src.roofline import (
    AttentionConfig,
    HardwareSpec,
    HARDWARE_SPECS,
    compute_arithmetic_intensity,
    compute_attention_flops,
    sweep_sequence_lengths,
)
from src.benchmark import BenchmarkSuite, estimate_memory_savings

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('paper', font_scale=1.2)
COLORS = {
    'fused': '#2196F3',      # Blue
    'naive': '#F44336',      # Red
    'pytorch': '#FF9800',    # Orange
    'ridge': '#4CAF50',      # Green
    'compute': '#9C27B0',    # Purple
    'memory': '#607D8B',     # Gray
}


def plot_roofline(
    hw: HardwareSpec,
    points: Optional[List[dict]] = None,
    seq_lengths: List[int] = None,
    batch: int = 4,
    num_heads: int = 32,
    head_dim: int = 64,
    save_path: Optional[str] = None,
):
    """
    Generate a roofline plot showing compute vs memory boundedness.

    The roofline model plots:
    - X-axis: Arithmetic intensity (FLOPS/byte)
    - Y-axis: Achieved/achievable FLOPS
    - Ceiling: min(peak_compute, peak_bandwidth × AI)

    Points below the roof show how far we are from hardware limits.

    Args:
        hw: Hardware specification
        points: Optional list of measured points with keys:
                'arithmetic_intensity', 'achieved_tflops', 'label'
        seq_lengths: Sequence lengths to plot theoretical points for
        batch, num_heads, head_dim: Config for theoretical analysis
        save_path: Path to save figure
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the roofline ceiling
    ai_range = np.logspace(-1, 4, 1000)
    achievable = np.array([hw.achievable_flops(ai) / 1e12 for ai in ai_range])
    ax.loglog(ai_range, achievable, 'k-', linewidth=3, label='Hardware Roofline')

    # Ridge point
    ridge = hw.ridge_point
    ridge_y = hw.peak_tflops_fp16
    ax.axvline(x=ridge, color=COLORS['ridge'], linestyle='--', alpha=0.7,
               label=f'Ridge Point ({ridge:.0f} FLOPS/byte)')

    # Add regime labels
    ax.text(ridge * 0.15, ridge_y * 0.6, 'MEMORY\nBOUND',
            fontsize=14, ha='center', color=COLORS['memory'], alpha=0.7, fontweight='bold')
    ax.text(ridge * 5, ridge_y * 0.6, 'COMPUTE\nBOUND',
            fontsize=14, ha='center', color=COLORS['compute'], alpha=0.7, fontweight='bold')

    # Plot theoretical points for fused attention at different seq lengths
    for seq_len in seq_lengths:
        config = AttentionConfig(batch_size=batch, num_heads=num_heads,
                                 seq_len=seq_len, head_dim=head_dim)

        # Fused
        ai_fused = compute_arithmetic_intensity(config, fused=True)
        peak_fused = hw.achievable_flops(ai_fused['arithmetic_intensity']) / 1e12
        ax.plot(ai_fused['arithmetic_intensity'], peak_fused, 'o',
                color=COLORS['fused'], markersize=8, zorder=5)
        ax.annotate(f'N={seq_len}', (ai_fused['arithmetic_intensity'], peak_fused),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

        # Naive
        ai_naive = compute_arithmetic_intensity(config, fused=False)
        peak_naive = hw.achievable_flops(ai_naive['arithmetic_intensity']) / 1e12
        ax.plot(ai_naive['arithmetic_intensity'], peak_naive, 's',
                color=COLORS['naive'], markersize=8, zorder=5)

    # Plot measured points if provided
    if points:
        for pt in points:
            ax.plot(pt['arithmetic_intensity'], pt['achieved_tflops'], '*',
                    color=COLORS['pytorch'], markersize=15, zorder=10)
            if 'label' in pt:
                ax.annotate(pt['label'],
                            (pt['arithmetic_intensity'], pt['achieved_tflops']),
                            textcoords="offset points", xytext=(8, -5), fontsize=8)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=3, label=f'{hw.name} Roofline'),
        Line2D([0], [0], color=COLORS['ridge'], linestyle='--',
               label=f'Ridge Point ({ridge:.0f} FLOP/byte)'),
        Line2D([0], [0], marker='o', color=COLORS['fused'], linestyle='',
               markersize=8, label='Fused Attention (FlashAttn)'),
        Line2D([0], [0], marker='s', color=COLORS['naive'], linestyle='',
               markersize=8, label='Naive Attention'),
    ]
    if points:
        legend_elements.append(
            Line2D([0], [0], marker='*', color=COLORS['pytorch'], linestyle='',
                   markersize=15, label='Measured Performance')
        )
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Labels
    ax.set_xlabel('Arithmetic Intensity (FLOPS / byte)', fontsize=13)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=13)
    ax.set_title(f'Roofline Model: Attention Kernel on {hw.name}\n'
                 f'(batch={batch}, heads={num_heads}, d={head_dim})', fontsize=14)

    # Format
    ax.set_xlim(0.5, 10000)
    ax.set_ylim(1, hw.peak_tflops_fp16 * 2)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_speedup_curve(
    suite: Optional[BenchmarkSuite] = None,
    seq_lengths: List[int] = None,
    theoretical: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot speedup of fused attention over naive across sequence lengths.

    Args:
        suite: BenchmarkSuite with measured results (optional)
        seq_lengths: Sequence lengths for theoretical analysis
        theoretical: Whether to plot theoretical speedup
        save_path: Path to save figure
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    fig, ax = plt.subplots(figsize=(10, 6))

    if theoretical:
        # Theoretical memory traffic ratio (speedup proxy)
        speedups = []
        for N in seq_lengths:
            config = AttentionConfig(batch_size=4, num_heads=32, seq_len=N, head_dim=64)
            ai_fused = compute_arithmetic_intensity(config, fused=True)
            ai_naive = compute_arithmetic_intensity(config, fused=False)
            # Speedup approximation: ratio of memory traffic
            # (since both versions do same FLOPS, speedup ~ naive_bytes / fused_bytes)
            speedup = ai_fused['arithmetic_intensity'] / ai_naive['arithmetic_intensity']
            speedups.append(speedup)

        ax.plot(seq_lengths, speedups, 'o-', color=COLORS['fused'],
                linewidth=2, markersize=8, label='Theoretical Speedup (memory traffic ratio)')

    # Measured speedups if available
    if suite and 'triton_flash' in suite.results and 'naive' in suite.results:
        measured_seq = []
        measured_speedups = []
        for t_res in suite.results['triton_flash']:
            for n_res in suite.results['naive']:
                if t_res.config.seq_len == n_res.config.seq_len:
                    measured_seq.append(t_res.config.seq_len)
                    measured_speedups.append(n_res.latency_ms / t_res.latency_ms)
        if measured_seq:
            ax.plot(measured_seq, measured_speedups, 's-', color=COLORS['naive'],
                    linewidth=2, markersize=8, label='Measured Speedup')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Breakeven')
    ax.set_xlabel('Sequence Length', fontsize=13)
    ax.set_ylabel('Speedup (Fused / Naive)', fontsize=13)
    ax.set_title('FlashAttention Speedup vs Sequence Length', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_memory_savings(
    seq_lengths: List[int] = None,
    batch: int = 4,
    num_heads: int = 32,
    head_dim: int = 64,
    save_path: Optional[str] = None,
):
    """
    Plot memory usage comparison: fused vs naive.

    Shows how memory grows with sequence length and where the N×N
    attention matrix dominates.

    Args:
        seq_lengths: Sequence lengths to analyze
        save_path: Path to save figure
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    naive_mem = []
    fused_mem = []
    attn_matrix_mem = []

    for N in seq_lengths:
        config = AttentionConfig(batch_size=batch, num_heads=num_heads,
                                 seq_len=N, head_dim=head_dim)
        mem = estimate_memory_savings(config)
        naive_mem.append(mem['naive_total_mb'])
        fused_mem.append(mem['fused_total_mb'])
        attn_matrix_mem.append(mem['attention_matrix_mb'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Absolute memory usage
    ax1.semilogy(seq_lengths, naive_mem, 'o-', color=COLORS['naive'],
                 linewidth=2, markersize=7, label='Naive (materializes N×N)')
    ax1.semilogy(seq_lengths, fused_mem, 's-', color=COLORS['fused'],
                 linewidth=2, markersize=7, label='Fused (FlashAttention)')
    ax1.semilogy(seq_lengths, attn_matrix_mem, '^--', color=COLORS['memory'],
                 linewidth=1.5, markersize=6, label='Attention Matrix alone', alpha=0.7)

    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Memory (MB, log scale)', fontsize=12)
    ax1.set_title('Memory Usage: Fused vs Naive', fontsize=13)
    ax1.set_xscale('log', base=2)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Savings percentage
    savings_pct = [(n - f) / n * 100 for n, f in zip(naive_mem, fused_mem)]
    ax2.plot(seq_lengths, savings_pct, 'o-', color=COLORS['fused'],
             linewidth=2, markersize=8)
    ax2.fill_between(seq_lengths, savings_pct, alpha=0.2, color=COLORS['fused'])
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Memory Savings (%)', fontsize=12)
    ax2.set_title('Memory Savings from Kernel Fusion', fontsize=13)
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_arithmetic_intensity_analysis(
    hw: HardwareSpec,
    seq_lengths: List[int] = None,
    save_path: Optional[str] = None,
):
    """
    Plot arithmetic intensity vs sequence length for fused and naive attention.

    Shows how fusion changes the compute/memory balance.
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    ai_fused_list = []
    ai_naive_list = []

    for N in seq_lengths:
        config = AttentionConfig(batch_size=4, num_heads=32, seq_len=N, head_dim=64)
        ai_f = compute_arithmetic_intensity(config, fused=True)
        ai_n = compute_arithmetic_intensity(config, fused=False)
        ai_fused_list.append(ai_f['arithmetic_intensity'])
        ai_naive_list.append(ai_n['arithmetic_intensity'])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(seq_lengths, ai_fused_list, 'o-', color=COLORS['fused'],
              linewidth=2, markersize=8, label='Fused (FlashAttention)')
    ax.loglog(seq_lengths, ai_naive_list, 's-', color=COLORS['naive'],
              linewidth=2, markersize=8, label='Naive (materializes N×N)')

    # Ridge point line
    ax.axhline(y=hw.ridge_point, color=COLORS['ridge'], linestyle='--',
               linewidth=2, label=f'Ridge Point ({hw.ridge_point:.0f} FLOP/byte)')

    # Shade regions
    ax.fill_between(seq_lengths, 0.1, hw.ridge_point, alpha=0.05, color=COLORS['memory'])
    ax.fill_between(seq_lengths, hw.ridge_point, 100000, alpha=0.05, color=COLORS['compute'])

    ax.text(200, hw.ridge_point * 0.3, 'Memory-Bound Region',
            fontsize=11, color=COLORS['memory'], alpha=0.8)
    ax.text(200, hw.ridge_point * 3, 'Compute-Bound Region',
            fontsize=11, color=COLORS['compute'], alpha=0.8)

    ax.set_xlabel('Sequence Length', fontsize=13)
    ax.set_ylabel('Arithmetic Intensity (FLOPS/byte)', fontsize=13)
    ax.set_title(f'Arithmetic Intensity vs Sequence Length ({hw.name})', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_all_plots(
    hw: HardwareSpec,
    suite: Optional[BenchmarkSuite] = None,
    measured_points: Optional[List[dict]] = None,
    save_dir: str = "results",
):
    """
    Generate all analysis plots.

    Args:
        hw: Hardware specification
        suite: Benchmark results (optional, for measured data)
        measured_points: Measured roofline points (optional)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Generating plots...")

    # 1. Roofline plot
    plot_roofline(hw, points=measured_points,
                  save_path=os.path.join(save_dir, 'roofline.png'))
    print(f"  Saved: roofline.png")

    # 2. Speedup curve
    plot_speedup_curve(suite=suite,
                       save_path=os.path.join(save_dir, 'speedup_curve.png'))
    print(f"  Saved: speedup_curve.png")

    # 3. Memory savings
    plot_memory_savings(save_path=os.path.join(save_dir, 'memory_savings.png'))
    print(f"  Saved: memory_savings.png")

    # 4. Arithmetic intensity analysis
    plot_arithmetic_intensity_analysis(hw,
        save_path=os.path.join(save_dir, 'arithmetic_intensity.png'))
    print(f"  Saved: arithmetic_intensity.png")

    print(f"\nAll plots saved to: {save_dir}/")


if __name__ == "__main__":
    """Generate theoretical analysis plots (no GPU needed)."""
    hw = HARDWARE_SPECS["A100_80GB"]
    generate_all_plots(hw, save_dir="/Users/jaikash/Desktop/triton-fused-attention/results")
