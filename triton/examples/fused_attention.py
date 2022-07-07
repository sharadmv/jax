from types import SimpleNamespace
import triton
import triton.language as tl
import jax_triton as jt

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

@triton.jit
def fused_attention_kernel(
    Q, K, V,
    TMP, L, M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    stride_qz, stride_qh, stride_qm, stride_qk = 196608, 65536, 64, 1
    stride_kz, stride_kh, stride_kk, stride_kn = 196608, 65536, 1024, 1
    stride_vz, stride_vh, stride_vk, stride_vn = 196608, 65536, 64, 1
    stride_oz, stride_oh, stride_om, stride_on = 196608, 65536, 64, 1
    Z, H, N_CTX = 2, 3, 1024
    start_qm = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_qm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * 1
    off_k = off_hz * stride_qh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * 1
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = TMP + off_hz * N_CTX + offs_m

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    q = tl.load(q_ptrs)
    for start_n in range(0, start_qm + 1):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        qk += tl.where(offs_m[:, None] >= (start_n * BLOCK_N + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        p = p.to(tl.float16)
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs)
        oo = tl.dot(p, v)
        acc += oo
        k_ptrs += BLOCK_N * 1
        v_ptrs += BLOCK_N * stride_vk
        # r_ptrs += BLOCK_N
        l_i = l_i_new
        m_i = m_i_new

    start_qm = tl.program_id(0)
    offs_m = start_qm * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_out = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * 1
    out_ptrs = Out + off_out
    tl.store(out_ptrs, acc)


def _strides(a):
  all = np.prod(a.shape)
  for s in a.shape:
    all = all // s
    yield all


def fused_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
  BLOCK = 128
  Lq, Lk = q.shape[-1], k.shape[-2]
  assert Lq == Lk
  grid = lambda _: (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
  out_shape = [
      SimpleNamespace(shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      SimpleNamespace(shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      SimpleNamespace(shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=q.dtype),
      SimpleNamespace(shape=q.shape, dtype=q.dtype)]
  metaparams = dict(
    BLOCK_M=BLOCK, BLOCK_DMODEL=64,
    BLOCK_N=BLOCK,
    num_warps=4, num_stages=1
  )
  _, _, _, output = jt.triton_call(q, k, v, kernel=fused_attention_kernel,
      out_shape=out_shape, grid=grid, **metaparams)
  return output


q_key, k_key, v_key = random.split(random.PRNGKey(0), 3)
q = random.normal(q_key, (2, 3, 1024, 64), dtype=jnp.float16)
k = random.normal(k_key, (2, 3, 64, 1024), dtype=jnp.float16)
v = random.normal(v_key, (2, 3, 1024, 64), dtype=jnp.float16)
print(fused_attention(q, k, v))
print(jax.jit(fused_attention)(q, k, v))
