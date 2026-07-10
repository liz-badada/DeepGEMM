"""KBC extra rows: dense D1x3/D2, masked-tiny G3, K-grouped K1, HyperConnection H1.

Reuses tests/generators.py so inputs are identical across checkouts (same seed).
Prints one TSV line per row: tag, shape_id, desc, latency_us, tflops, gb_s, diff.
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
    KernelType, MajorTypeAB, QuantConfig, generate_normal,
    generate_m_grouped_masked, generate_k_grouped_contiguous,
    get_mk_alignment_for_contiguous_layout, get_ue8m0_usage, layout_masked_to_psum,
)

TAG = sys.argv[1] if len(sys.argv) > 1 else 'x'
KT = KernelType.Kernel1D1D
QC_FP8 = QuantConfig((128, 128, False, False))
QC_FP4 = QuantConfig((32, 32, True, True))
QC_MIX = QuantConfig((128, 32, False, True))


def emit(shape_id: str, desc: str, t: float, tflops: float, gb_s: float, diff: float) -> None:
    print(f'{TAG}\t{shape_id}\t{desc}\t{t * 1e6:.1f}\t{tflops:.1f}\t{gb_s:.1f}\t{diff:.5f}', flush=True)


def reseed(salt: int) -> None:
    torch.manual_seed(1234 + salt)
    random.seed(1234 + salt)


def bench_dense(shape_id: str, qc: QuantConfig, m: int, n: int, k: int, salt: int) -> None:
    reseed(salt)
    use_ue8m0 = get_ue8m0_usage(KT)
    recipe, recipe_a, recipe_b = qc.get_recipes(is_wgrad=False)
    a, b, c, d, ref_d = generate_normal(
        m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False, torch.bfloat16, KT,
        use_ue8m0=use_ue8m0, quant_config=qc)

    def fn() -> None:
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=not use_ue8m0,
                                  recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

    fn()
    diff = calc_diff(d, ref_d)
    assert diff < qc.max_diff(), f'{shape_id}: {diff}'
    t = bench_kineto(fn, 'gemm_', suppress_kineto_output=True)
    emit(shape_id, f'm{m}_n{n}_k{k}_NT', t, 2 * m * n * k / t / 1e12, count_bytes(a, b, d) / t / 1e9, diff)


def bench_g3_masked_psum(salt: int) -> None:
    reseed(salt)
    num_groups, max_m, expected_m, n, k = 6, 4096, 20, 4096, 2048
    qc = QC_FP4
    use_ue8m0 = get_ue8m0_usage(KT)
    recipe, recipe_a, recipe_b = qc.get_recipes()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(expected_m * 1.2))
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    num_tests = 8
    sum_t, sum_ops, sum_bytes, worst_diff = 0.0, 0.0, 0.0, 0.0
    for _ in range(num_tests):
        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m, n, k,
            use_ue8m0=use_ue8m0, use_psum_layout=True, quant_config=qc)
        a_psum = (layout_masked_to_psum(a[0], psum_m), layout_masked_to_psum(a[1], psum_m))
        d_psum = layout_masked_to_psum(d, psum_m)

        def fn() -> None:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a_psum, b, d_psum, psum_m, disable_ue8m0_cast=not use_ue8m0,
                use_psum_layout=True, expected_m_for_psum_layout=int(expected_m * 1.2),
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

        fn()
        for j in range(num_groups):
            if masked_m[j].item() == 0:
                continue
            d_slice = d_psum[: psum_m[j]] if j == 0 else \
                d_psum[align(psum_m[j - 1], get_mk_alignment_for_contiguous_layout()): psum_m[j]]
            diff = calc_diff(d_slice, ref_d[j, :masked_m[j].item()])
            assert diff < qc.max_diff(), f'G3 group {j}: {diff}'
            worst_diff = max(worst_diff, diff)

        valid_m = masked_m.sum().item()
        t = bench_kineto(fn, 'gemm_', suppress_kineto_output=True)
        sum_t += t
        sum_ops += 2 * valid_m * n * k
        sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)
    t = sum_t / num_tests
    emit('G3', f'g{num_groups}_em{expected_m}_n{n}_k{k}_psum1',
         t, sum_ops / sum_t / 1e12, sum_bytes / sum_t / 1e9, worst_diff)


def bench_k1_kgrouped(salt: int) -> None:
    reseed(salt)
    num_groups, m, n, expected_k, gran_k = 4, 4096, 7168, 8192, 128
    deep_gemm.set_mk_alignment_for_contiguous_layout(gran_k)
    ks = [align(int(expected_k * random.uniform(0.7, 1.3)),
                get_mk_alignment_for_contiguous_layout()) for _ in range(num_groups)]
    use_ue8m0 = get_ue8m0_usage(KT)
    k, a, b, c, d, ref_d = generate_k_grouped_contiguous(
        num_groups, m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
        use_ue8m0=use_ue8m0, gran_k=gran_k)
    ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')

    def fn() -> None:
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, ks_tensor, c, recipe=(1, 1, gran_k))

    fn()
    diff = calc_diff(d, ref_d)
    assert diff < 0.001, f'K1: {diff}'
    t = bench_kineto(fn, 'gemm_', suppress_kineto_output=True)
    emit('K1', f'g{num_groups}_m{m}_n{n}_k{k}_gran{gran_k}',
         t, 2 * m * n * k / t / 1e12, count_bytes(a, b, c, d) / t / 1e9, diff)


def bench_h1_hyperconnection(salt: int) -> None:
    reseed(salt)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    m, n, k = 8192, 24, 28672
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.float, device='cuda')
    d = torch.empty((m, n), dtype=torch.float, device='cuda')
    s = torch.empty((m,), dtype=torch.float, device='cuda')
    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=None)
    ref_d = a.float() @ b.T
    ref_s = a.float().square().sum(-1)
    diff = max(calc_diff(d, ref_d), calc_diff(s, ref_s))
    assert diff < 1e-8, f'H1: {diff}'
    t = bench_kineto(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=None),
                     'tf32_hc_prenorm_gemm', suppress_kineto_output=True)
    emit('H1', f'm{m}_n{n}_k{k}_splits0',
         t, 2 * m * n * k / t / 1e12, count_bytes(a, b, d, s) / t / 1e9, diff)


def main() -> None:
    bench_dense('D1_FP8', QC_FP8, 4096, 16384, 7168, salt=1)
    bench_dense('D1_FP4', QC_FP4, 4096, 16384, 7168, salt=2)
    bench_dense('D1_MIX', QC_MIX, 4096, 16384, 7168, salt=3)
    bench_dense('D2_FP4', QC_FP4, 128, 24576, 1536, salt=4)
    bench_g3_masked_psum(salt=5)
    bench_k1_kgrouped(salt=6)
    bench_h1_hyperconnection(salt=7)


if __name__ == '__main__':
    main()
