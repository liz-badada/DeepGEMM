"""KBC attention rows: A1 regular MQA (FP4/FP8), A2/A3 paged MQA (varlen, MaxTPB=1).

Fixed-configuration replicas of the tests/test_attention.py case bodies so the
same inputs (same seed) run on any checkout. TSV per row:
tag, shape_id, desc, latency_us, tflops, gb_s, diff.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8, per_token_cast_to_fp4, cast_back_from_fp4
from test_attention import ref_fp8_mqa_logits, ref_paged_mqa_logits

TAG = sys.argv[1] if len(sys.argv) > 1 else 'x'


def emit(shape_id: str, desc: str, t: float, tflops: float, gb_s: float, diff: float) -> None:
    print(f'{TAG}\t{shape_id}\t{desc}\t{t * 1e6:.1f}\t{tflops:.1f}\t{gb_s:.1f}\t{diff:.5f}', flush=True)


def reseed(salt: int) -> None:
    torch.manual_seed(9876 + salt)
    random.seed(9876 + salt)


def bench_a1_mqa(shape_id: str, is_fp4: bool, logits_dtype: torch.dtype, salt: int) -> None:
    reseed(salt)
    seq_len, seq_len_kv, num_heads, head_dim = 4096, 8192, 64, 128

    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
    # disable_cp=True (CP=0)
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)

    ref_logits, ref_cost = ref_fp8_mqa_logits(q, kv, weights, ks, ke)

    if is_fp4:
        q_fp4 = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        q_in = (q_fp4[0].view(seq_len, num_heads, head_dim // 2), q_fp4[1].view(seq_len, num_heads))
        kv_fp4 = per_token_cast_to_fp4(kv.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        kv_in = (kv_fp4[0].view(seq_len_kv, head_dim // 2), kv_fp4[1].view(seq_len_kv))
    else:
        q_in = q.to(torch.float8_e4m3fn), None
        kv_in = per_custom_dims_cast_to_fp8(kv, (0,), False)

    kernel_kwargs = dict(
        q=q_in, kv=kv_in, weights=weights,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
        clean_logits=True, max_seqlen_k=0,
        logits_dtype=logits_dtype)

    logits = deep_gemm.fp8_fp4_mqa_logits(**kernel_kwargs)

    ref_neginf_mask = (ref_logits == float('-inf'))
    assert torch.equal(logits == float('-inf'), ref_neginf_mask)
    diff = calc_diff(logits.masked_fill(ref_neginf_mask, 0),
                     ref_logits.masked_fill(ref_neginf_mask, 0))
    assert diff < (0.02 if is_fp4 else 1e-3), f'{shape_id}: {diff}'

    tflops_total = 2 * ref_cost * num_heads * head_dim / 1e12
    t, _ = bench_kineto(lambda: deep_gemm.fp8_fp4_mqa_logits(**kernel_kwargs),
                        ('mqa_logits', 'clean_logits'))
    gb_s = (count_bytes(q_in, kv_in, weights, ks, ke) + ref_cost * 4) / t / 1e9
    emit(shape_id, f's{seq_len}_skv{seq_len_kv}_h{num_heads}_d{head_dim}_cp0',
         t, tflops_total / t, gb_s, diff)


def _kv_cache_cast_to_fp8(x: torch.Tensor):
    num_blocks, block_size, num_heads, head_dim = x.shape
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_cast_back = x_scaled.float() * sf
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), x_cast_back.to(x.dtype)


def _kv_cache_cast_to_fp4(x: torch.Tensor):
    num_blocks, block_size, num_heads, head_dim = x.shape
    x_scaled, sf = per_token_cast_to_fp4(x.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    x_cast_back = cast_back_from_fp4(x_scaled, sf, gran_k=32, use_packed_ue8m0=True).view(num_blocks, block_size, 1, head_dim)
    x_fp4 = torch.empty((num_blocks, block_size * (head_dim // 2 + 4)), device=x.device, dtype=torch.uint8)
    x_fp4[:, : block_size * head_dim // 2] = x_scaled.view(num_blocks, block_size * head_dim // 2).view(torch.uint8)
    x_fp4[:, block_size * head_dim // 2:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp4.view(num_blocks, block_size, num_heads, head_dim // 2 + 4), x_cast_back.to(x.dtype)


def bench_paged(shape_id: str, is_fp4: bool, logits_dtype: torch.dtype, avg_kv: int, salt: int) -> None:
    # Fixed config: is_varlen=True, MaxTPB=1, batch=256, next_n=1, block_kv=64,
    # use_2d_context_lens=True, clean_logits=False, H=64, D=128.
    reseed(salt)
    block_kv, batch_size, next_n, max_tokens_per_batch = 64, 256, 1, 1
    num_heads, head_dim = 64, 128
    max_model_len = 111 * 1024
    min_blocks_needed = batch_size * (int(1.3 * avg_kv) + max_tokens_per_batch) // block_kv + 256
    num_total_blocks = max(min_blocks_needed, max_model_len) * (1 if is_fp4 else 5)

    raw_batch_size = batch_size
    tokens_per_seq = torch.randint(1, max_tokens_per_batch + 1, (raw_batch_size,), device='cuda', dtype=torch.int)
    indices = torch.arange(raw_batch_size, device='cuda', dtype=torch.int).repeat_interleave(tokens_per_seq)
    batch_size, next_n = tokens_per_seq.sum().item(), 1

    q = torch.randn((batch_size, next_n, num_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    kv_cache = torch.randn((num_total_blocks, block_kv, 1, head_dim), device='cuda', dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, num_heads), device='cuda', dtype=torch.float)
    context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (raw_batch_size,), device='cuda', dtype=torch.int)
    max_ctx_len_per_seq = context_lens + (tokens_per_seq - 1)

    seq_sum_lens = context_lens.sum().item()
    num_blocks_per_query = ceil_div(max_ctx_len_per_seq, block_kv)
    block_table = torch.empty((raw_batch_size, num_blocks_per_query.max().item()), device='cuda', dtype=torch.int)
    block_idx_pool = torch.randperm(num_total_blocks, device='cuda', dtype=torch.int)
    offset = 0
    for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
        block_table[i, :num_blocks] = block_idx_pool[offset: offset + num_blocks]
        offset += num_blocks
    context_lens = context_lens.repeat_interleave(tokens_per_seq)
    offsets_within_seq = torch.cat([
        torch.arange(n.item(), device='cuda', dtype=torch.int) for n in tokens_per_seq])
    context_lens = context_lens + offsets_within_seq
    block_table = block_table.repeat_interleave(tokens_per_seq, dim=0)

    ref_logits = ref_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, max_model_len, True)

    if is_fp4:
        q_fp4 = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        q_in = (q_fp4[0].view(batch_size, next_n, num_heads, head_dim // 2),
                q_fp4[1].view(batch_size, next_n, num_heads))
        kv_in, _ = _kv_cache_cast_to_fp4(kv_cache)
    else:
        q_in = q.to(torch.float8_e4m3fn), None
        kv_in, _ = _kv_cache_cast_to_fp8(kv_cache)

    positions = torch.arange(max_model_len, device='cuda').unsqueeze(0).expand(batch_size * next_n, -1)
    context_lens_nextn = context_lens.view(-1, 1)
    ref_neginf_mask = ~(positions < context_lens_nextn.view(-1, 1))

    num_clusters = deep_gemm.get_num_sms()
    kernel_kwargs = dict(
        q=q_in, kv_cache=kv_in, weights=weights,
        context_lens=context_lens_nextn, block_table=block_table,
        schedule_meta=deep_gemm.get_paged_mqa_logits_metadata(context_lens_nextn, block_kv, num_clusters, indices=indices),
        max_context_len=max_model_len, clean_logits=False, logits_dtype=logits_dtype,
        indices=indices)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(**kernel_kwargs).to(torch.float)

    diff = calc_diff(logits.masked_fill(ref_neginf_mask, 0),
                     ref_logits.masked_fill(ref_neginf_mask, 0))
    assert diff < (0.02 if is_fp4 else 1e-3), f'{shape_id}: {diff}'

    sum_lens = context_lens.sum().item()
    tflops_total = 2 * sum_lens * next_n * num_heads * head_dim / 1e12
    kv_bytes_per_token = head_dim / (2 if is_fp4 else 1) + 4
    total_bytes = count_bytes(q, weights) + seq_sum_lens * kv_bytes_per_token + (sum_lens * next_n * logits_dtype.itemsize)

    t = bench_kineto(lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**kernel_kwargs), 'paged_mqa_logits')
    emit(shape_id, f'bkv{block_kv}_b{raw_batch_size}_n1_h{num_heads}_d{head_dim}_L{avg_kv}_mtpb1',
         t, tflops_total / t, total_bytes / t / 1e9, diff)


def main() -> None:
    bench_a1_mqa('A1_FP4', True, torch.bfloat16, salt=1)
    bench_a1_mqa('A1_FP8', False, torch.float, salt=2)
    bench_paged('A2_FP4', True, torch.bfloat16, 32768, salt=3)
    bench_paged('A3_FP8', False, torch.float, 8192, salt=4)


if __name__ == '__main__':
    main()
