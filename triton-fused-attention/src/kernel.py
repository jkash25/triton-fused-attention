"""
Triton Fused Attention Kernel — FlashAttention-style tiled implementation.

This implements the FlashAttention algorithm in Triton:
- Tiles Q, K, V into SRAM-sized blocks to avoid materializing the full N×N
  attention matrix in HBM
- Fuses softmax + matmul into a single kernel pass
- Uses online softmax (log-sum-exp trick) for numerical stability
- Supports optional causal masking

Memory complexity: O(N) instead of O(N²) for the attention matrix
Compute complexity: O(N²d) — same as standard attention, but with fewer HBM accesses

Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness" (2022)
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _fused_attention_fwd_kernel(
    # Pointers to input tensors
    Q_ptr, K_ptr, V_ptr,
    # Pointer to output tensor
    O_ptr,
    # Pointer to log-sum-exp (for backward pass / debugging)
    LSE_ptr,
    # Tensor dimensions
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    # Strides for Q (batch*heads, seq_len, head_dim)
    stride_qb, stride_qm, stride_qk,
    # Strides for K
    stride_kb, stride_kn, stride_kk,
    # Strides for V
    stride_vb, stride_vn, stride_vk,
    # Strides for O
    stride_ob, stride_om, stride_ok,
    # Stride for LSE (batch*heads, seq_len)
    stride_lse_b, stride_lse_m,
    # Scaling factor (1/sqrt(d))
    sm_scale,
    # Causal masking flag
    IS_CAUSAL: tl.constexpr,
    # Block sizes (SRAM tile dimensions)
    BLOCK_M: tl.constexpr,  # Block size for query sequence dimension
    BLOCK_N: tl.constexpr,  # Block size for key/value sequence dimension
    BLOCK_D: tl.constexpr,  # Block size for head dimension (must be >= head_dim)
):
    """
    FlashAttention forward pass kernel.

    Each program instance computes one BLOCK_M × head_dim tile of the output.
    It iterates over all BLOCK_N-sized blocks of K and V, accumulating the
    attention-weighted sum using online softmax.

    Grid: (num_blocks_m, batch * num_heads)
    """
    # Program ID: which block of queries and which batch/head
    block_m_idx = tl.program_id(0)  # Which query block
    batch_head_idx = tl.program_id(1)  # Which batch × head

    # Compute offsets for this query block
    # Rows of Q this block is responsible for
    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    # Columns (head dimension)
    offs_d = tl.arange(0, BLOCK_D)
    # Key/value sequence positions (iterated in the inner loop)
    offs_n = tl.arange(0, BLOCK_N)

    # Base pointers for this batch/head
    q_base = Q_ptr + batch_head_idx * stride_qb
    k_base = K_ptr + batch_head_idx * stride_kb
    v_base = V_ptr + batch_head_idx * stride_vb
    o_base = O_ptr + batch_head_idx * stride_ob
    lse_base = LSE_ptr + batch_head_idx * stride_lse_b

    # Load Q block: [BLOCK_M, BLOCK_D]
    # Mask for valid query positions (handles seq_len not divisible by BLOCK_M)
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators for online softmax
    # m_i: running max of (Q @ K^T) * scale for numerical stability
    # l_i: running sum of exp(x - m_i) for normalization
    # o_i: running weighted sum (output accumulator)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Determine the range of K/V blocks to iterate over
    if IS_CAUSAL:
        # For causal attention, we only need K[:block_m_idx*BLOCK_M + BLOCK_M]
        num_blocks_n = tl.cdiv(block_m_idx * BLOCK_M + BLOCK_M, BLOCK_N)
    else:
        num_blocks_n = tl.cdiv(seq_len, BLOCK_N)

    # Main loop: iterate over K/V blocks
    for block_n_idx in range(0, num_blocks_n):
        # Current key positions
        curr_offs_n = block_n_idx * BLOCK_N + offs_n

        # Load K block: [BLOCK_N, BLOCK_D] -> we need [BLOCK_D, BLOCK_N] for matmul
        k_mask = (curr_offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k_ptrs = k_base + curr_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute attention scores: S = Q @ K^T * scale
        # q: [BLOCK_M, BLOCK_D], k^T: [BLOCK_D, BLOCK_N]
        # s: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # Apply causal mask: positions where query_idx < key_idx get -inf
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= curr_offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Mask out-of-bounds key positions
        s = tl.where(curr_offs_n[None, :] < seq_len, s, float("-inf"))

        # Online softmax update
        # New maximum for this block
        m_ij = tl.max(s, axis=1)  # [BLOCK_M]
        # New maximum across all blocks so far
        m_new = tl.maximum(m_i, m_ij)
        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)
        # Exponentials of current block scores
        p = tl.exp(s - m_new[:, None])  # [BLOCK_M, BLOCK_N]

        # Update running sum of exponentials
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # Load V block: [BLOCK_N, BLOCK_D]
        v_mask = (curr_offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v_ptrs = v_base + curr_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Update output accumulator
        # Rescale previous output by correction factor
        o_i = o_i * alpha[:, None]
        # Add contribution from current block: P @ V
        o_i += tl.dot(p.to(v.dtype), v)

        # Update running statistics
        m_i = m_new
        l_i = l_new

    # Final normalization: divide by sum of exponentials
    o_i = o_i / l_i[:, None]

    # Store output
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, o_i.to(tl.float16), mask=o_mask)

    # Store log-sum-exp for potential backward pass
    lse_mask = offs_m < seq_len
    lse_ptrs = lse_base + offs_m * stride_lse_m
    lse = m_i + tl.log(l_i)
    tl.store(lse_ptrs, lse, mask=lse_mask)


class TritonFlashAttention:
    """
    FlashAttention-style fused attention using Triton.

    Computes: softmax(Q @ K^T / sqrt(d)) @ V
    in a single fused kernel with O(N) memory instead of O(N²).

    Args:
        causal: Whether to apply causal (autoregressive) masking.
        block_m: Tile size for query sequence dimension.
        block_n: Tile size for key/value sequence dimension.
    """

    def __init__(self, causal: bool = False, block_m: int = 64, block_n: int = 64):
        self.causal = causal
        self.block_m = block_m
        self.block_n = block_n

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            v: Value tensor [batch, num_heads, seq_len, head_dim]

        Returns:
            Output tensor [batch, num_heads, seq_len, head_dim]
        """
        return flash_attention_forward(q, k, v, causal=self.causal,
                                       block_m=self.block_m, block_n=self.block_n)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    Compute FlashAttention forward pass using the Triton kernel.

    Args:
        q: [batch, num_heads, seq_len, head_dim] — query tensor (fp16)
        k: [batch, num_heads, seq_len, head_dim] — key tensor (fp16)
        v: [batch, num_heads, seq_len, head_dim] — value tensor (fp16)
        causal: Whether to apply causal masking
        block_m: Query block size (must be power of 2)
        block_n: Key/Value block size (must be power of 2)

    Returns:
        o: [batch, num_heads, seq_len, head_dim] — output tensor (fp16)
    """
    assert q.dim() == 4, f"Expected 4D tensor, got {q.dim()}D"
    assert q.dtype == torch.float16, f"Expected fp16, got {q.dtype}"

    batch, num_heads, seq_len, head_dim = q.shape
    assert k.shape == q.shape and v.shape == q.shape

    # Scaling factor
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Reshape to (batch * num_heads, seq_len, head_dim) for the kernel
    q_flat = q.reshape(batch * num_heads, seq_len, head_dim).contiguous()
    k_flat = k.reshape(batch * num_heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * num_heads, seq_len, head_dim).contiguous()

    # Allocate output
    o_flat = torch.empty_like(q_flat)

    # Allocate log-sum-exp buffer
    lse = torch.empty(batch * num_heads, seq_len, device=q.device, dtype=torch.float32)

    # Compute block dimension (must be >= head_dim and power of 2)
    block_d = triton.next_power_of_2(head_dim)

    # Grid: (number of query blocks, batch * num_heads)
    num_blocks_m = triton.cdiv(seq_len, block_m)
    grid = (num_blocks_m, batch * num_heads)

    # Launch kernel
    _fused_attention_fwd_kernel[grid](
        q_flat, k_flat, v_flat,
        o_flat,
        lse,
        seq_len=seq_len,
        head_dim=head_dim,
        stride_qb=q_flat.stride(0), stride_qm=q_flat.stride(1), stride_qk=q_flat.stride(2),
        stride_kb=k_flat.stride(0), stride_kn=k_flat.stride(1), stride_kk=k_flat.stride(2),
        stride_vb=v_flat.stride(0), stride_vn=v_flat.stride(1), stride_vk=v_flat.stride(2),
        stride_ob=o_flat.stride(0), stride_om=o_flat.stride(1), stride_ok=o_flat.stride(2),
        stride_lse_b=lse.stride(0), stride_lse_m=lse.stride(1),
        sm_scale=sm_scale,
        IS_CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )

    # Reshape back to (batch, num_heads, seq_len, head_dim)
    o = o_flat.reshape(batch, num_heads, seq_len, head_dim)
    return o


def get_kernel_info():
    """Return a description of the kernel configuration for reporting."""
    return {
        "name": "TritonFlashAttention",
        "algorithm": "FlashAttention (Dao et al., 2022)",
        "implementation": "Triton @triton.jit kernel",
        "memory_complexity": "O(N) — no materialized attention matrix",
        "compute_complexity": "O(N²d) — same as standard attention",
        "key_techniques": [
            "Tiled computation (Q/K/V blocked into SRAM-sized tiles)",
            "Online softmax (numerically stable, single-pass)",
            "Fused softmax + matmul (avoids HBM round-trip)",
            "Causal masking support (early termination of K/V loop)",
        ],
    }
