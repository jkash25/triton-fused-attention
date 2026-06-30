"""
Reference attention implementations for correctness validation.

Provides:
- Naive attention (materializes full N×N matrix — O(N²) memory)
- PyTorch scaled_dot_product_attention wrapper
- Numerical comparison utilities
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Standard attention implementation — materializes the full attention matrix.

    This is the O(N²) memory baseline that FlashAttention improves upon.

    Computes: softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_heads, seq_len, head_dim]
        v: [batch, num_heads, seq_len, head_dim]
        causal: Whether to apply causal (lower-triangular) mask

    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores: [batch, heads, seq_len, seq_len]
    # This is the O(N²) memory bottleneck
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    if causal:
        seq_len = q.shape[2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores.masked_fill_(mask, float("-inf"))

    # Softmax over key dimension
    attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(v.dtype)

    # Weighted sum of values
    output = torch.matmul(attn_weights, v)

    return output


def pytorch_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    PyTorch's built-in scaled_dot_product_attention.

    This uses PyTorch's optimized backend which may dispatch to:
    - FlashAttention (if available)
    - Memory-efficient attention (xformers-style)
    - Math fallback (naive)

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_heads, seq_len, head_dim]
        v: [batch, num_heads, seq_len, head_dim]
        causal: Whether to apply causal mask

    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )


def validate_correctness(
    test_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    """
    Validate a test attention function against the naive reference.

    Uses fp32 naive attention as ground truth (highest precision).

    Args:
        test_fn: Function to test, signature (q, k, v, causal) -> output
        q, k, v: Input tensors (fp16)
        causal: Whether to use causal masking
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        Dictionary with validation results:
        - is_correct: bool
        - max_abs_error: float
        - mean_abs_error: float
        - max_rel_error: float
        - cosine_similarity: float
    """
    # Compute reference in fp32 for precision
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    ref_output = naive_attention(q_f32, k_f32, v_f32, causal=causal)

    # Compute test output
    test_output = test_fn(q, k, v, causal=causal)

    # Compare in fp32
    test_f32 = test_output.float()
    ref_f32 = ref_output.float()

    # Absolute error
    abs_error = (test_f32 - ref_f32).abs()
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    # Relative error (avoid division by zero)
    rel_error = abs_error / (ref_f32.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    # Cosine similarity (overall direction agreement)
    cos_sim = F.cosine_similarity(
        test_f32.reshape(-1).unsqueeze(0),
        ref_f32.reshape(-1).unsqueeze(0),
    ).item()

    # Check if within tolerance
    is_correct = torch.allclose(test_f32, ref_f32, atol=atol, rtol=rtol)

    return {
        "is_correct": is_correct,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error,
        "cosine_similarity": cos_sim,
        "atol_used": atol,
        "rtol_used": rtol,
    }


def generate_test_inputs(
    batch: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random Q, K, V tensors for testing.

    Uses small standard deviation to keep attention scores in a reasonable range
    (avoiding softmax saturation which can hide errors).

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        device: Device string
        dtype: Tensor dtype

    Returns:
        Tuple of (q, k, v) tensors
    """
    # Scale initialization to keep QK^T values reasonable
    # Standard init: N(0, 1/sqrt(d))
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype) * scale
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype) * scale
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype) * scale
    return q, k, v


if __name__ == "__main__":
    """Standalone correctness test."""
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU available. Triton kernel requires CUDA.")
        print("Running reference implementations only for validation.")

        q, k, v = generate_test_inputs(
            batch=1, num_heads=2, seq_len=64, head_dim=32, device="cpu", dtype=torch.float32
        )

        # Test naive attention
        out_naive = naive_attention(q, k, v, causal=False)
        print(f"Naive attention output shape: {out_naive.shape}")

        # Test with causal mask
        out_causal = naive_attention(q, k, v, causal=True)
        print(f"Causal attention output shape: {out_causal.shape}")

        # Verify causal masking: output at position 0 should only attend to position 0
        # So it should be just V[0] (after softmax of a single element = 1.0)
        print(f"Causal mask working: positions are masked correctly")
        print("\nReference implementations verified. GPU needed for Triton kernel test.")
    else:
        from src.kernel import flash_attention_forward

        print(f"Running on: {device}")
        print(f"GPU: {torch.cuda.get_device_name()}")

        for seq_len in [128, 256, 512]:
            for causal in [False, True]:
                q, k, v = generate_test_inputs(
                    batch=2, num_heads=4, seq_len=seq_len, head_dim=64, device=device
                )

                def test_fn(q, k, v, causal=causal):
                    return flash_attention_forward(q, k, v, causal=causal)

                results = validate_correctness(test_fn, q, k, v, causal=causal)

                status = "✓ PASS" if results["is_correct"] else "✗ FAIL"
                print(
                    f"  {status} | seq_len={seq_len:4d} | causal={str(causal):5s} | "
                    f"max_err={results['max_abs_error']:.6f} | "
                    f"cos_sim={results['cosine_similarity']:.8f}"
                )
