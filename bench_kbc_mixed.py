"""Serving-dtype (W4A8: FP8 act 1x128 x FP4 weight 1x32) G1/G2 benchmark.

Mirrors the SGLang DSv4 MoE runner recipes (recipe_a=(1,128), recipe_b=(1,32)):
G1 = m-grouped contiguous (m_indices, non-psum), G2 = m-grouped masked.
TSV per row: tag, shape_id, desc, latency_us, tflops, gb_s, diff.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import align
from generators import (
    KernelType, MajorTypeAB, QuantConfig,
    generate_m_grouped_contiguous, generate_m_grouped_masked,
    get_mk_alignment_for_contiguous_layout, get_ue8m0_usage,
)

TAG = sys.argv[1] if len(sys.argv) > 1 else 'x'
KT = KernelType.Kernel1D1D
QC_MIX = QuantConfig((128, 32, False, True))  # FP8 act (1,128) x FP4 weight (1,32)


def emit(shape_id: str, desc: str, t: float, tflops: float, gb_s: float, diff: float) -> None:
    print(f'{TAG}\t{shape_id}\t{desc}\t{t * 1e6:.1f}\t{tflops:.1f}\t{gb_s:.1f}\t{diff:.5f}', flush=True)


def reseed(salt: int) -> None:
    torch.manual_seed(5150 + salt)
    random.seed(5150 + salt)


def bench_g1_mixed(salt: int) -> None:
    reseed(salt)
    num_groups, expected_m, n, k = 4, 8192, 6144, 7168
    use_ue8m0 = get_ue8m0_usage(KT)
    recipe, recipe_a, recipe_b = QC_MIX.get_recipes()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=use_ue8m0, use_psum_layout=False, quant_config=QC_MIX)

    def fn() -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, b, d, grouped_layout, disable_ue8m0_cast=not use_ue8m0,
            use_psum_layout=False, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

    fn()
    diff = calc_diff(d, ref_d)
    assert diff < QC_MIX.max_diff(), f'G1_MIX: {diff}'
    t = bench_kineto(fn, 'gemm_', suppress_kineto_output=True)
    emit('G1_MIX', f'g{num_groups}_m{m}_n{n}_k{k}_contig_w4a8',
         t, 2 * m * n * k / t / 1e12, count_bytes(a, b, d) / t / 1e9, diff)


def bench_g2_mixed(salt: int) -> None:
    reseed(salt)
    num_groups, max_m, expected_m, n, k = 6, 4096, 1024, 4096, 4096
    use_ue8m0 = get_ue8m0_usage(KT)
    recipe, recipe_a, recipe_b = QC_MIX.get_recipes()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(expected_m * 1.2))
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    num_tests = 8
    sum_t, sum_ops, sum_bytes, worst_diff = 0.0, 0.0, 0.0, 0.0
    for _ in range(num_tests):
        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m, n, k,
            use_ue8m0=use_ue8m0, use_psum_layout=False, quant_config=QC_MIX)

        def fn() -> None:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
                a, b, d, masked_m, int(expected_m * 1.2), disable_ue8m0_cast=not use_ue8m0,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

        fn()
        for j in range(num_groups):
            if masked_m[j].item() == 0:
                continue
            diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
            assert diff < QC_MIX.max_diff(), f'G2_MIX group {j}: {diff}'
            worst_diff = max(worst_diff, diff)

        valid_m = masked_m.sum().item()
        t = bench_kineto(fn, 'gemm_', suppress_kineto_output=True)
        sum_t += t
        sum_ops += 2 * valid_m * n * k
        sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)
    t = sum_t / num_tests
    emit('G2_MIX', f'g{num_groups}_em{expected_m}_n{n}_k{k}_masked_w4a8',
         t, sum_ops / sum_t / 1e12, sum_bytes / sum_t / 1e9, worst_diff)


def main() -> None:
    bench_g1_mixed(salt=1)
    bench_g2_mixed(salt=2)


if __name__ == '__main__':
    main()
