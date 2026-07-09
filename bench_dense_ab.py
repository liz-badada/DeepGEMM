"""Dense-only A/B bench: reuses tests/generators.py so results are comparable
across checkouts. Prints one TSV line per case: tag, quant, m, n, k, us, diff."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff
from generators import (
    MajorTypeAB, enumerate_normal, generate_normal, get_ue8m0_usage,
)

TAG = sys.argv[1] if len(sys.argv) > 1 else 'x'
SHAPES = {(1, 4096, 7168), (128, 4096, 7168), (4096, 4096, 7168), (4096, 7168, 2048)}


def main() -> None:
    for kernel_type, qc, m, n, k, major_a, major_b, acc, out_dtype in enumerate_normal(torch.float8_e4m3fn):
        if (m, n, k) not in SHAPES or acc or out_dtype != torch.bfloat16:
            continue
        if not (major_a.is_k_major() and major_b.is_k_major()):
            continue
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        recipe, recipe_a, recipe_b = qc.get_recipes(is_wgrad=False)
        a, b, c, d, ref_d = generate_normal(
            m, n, k, major_a, major_b, acc, out_dtype, kernel_type,
            use_ue8m0=use_ue8m0, quant_config=qc)

        def fn() -> None:
            deep_gemm.fp8_fp4_gemm_nt(
                a, b, d, c=c, disable_ue8m0_cast=not use_ue8m0,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

        fn()
        diff = calc_diff(d, ref_d)
        t = bench_kineto(fn, 'gemm_', suppress_kineto_output=True)
        fp4_a = int(bool(getattr(qc, 'is_fp4_a', False)))
        fp4_b = int(bool(getattr(qc, 'is_fp4_b', False)))
        qname = f'fp4a{fp4_a}b{fp4_b}_ra{recipe_a}_rb{recipe_b}'
        print(f'{TAG}\t{qname}\t{m}\t{n}\t{k}\t{t * 1e6:.1f}\t{diff:.5f}', flush=True)


if __name__ == '__main__':
    main()
