"""Tile sweep for DSv4 dense Normal GEMM small-M AB-swap path."""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

from bench_dsv4_dense_normal_shapes import (
    DEFAULT_DECODE_M_VALUES,
    DENSE_NK_SHAPES,
    expected_runtime_shape,
    load_gpu_modules,
    make_output_path,
    parse_int_list,
    quant_config_from_name,
    quant_names_from_arg,
    selected_shapes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sweep SM120 dense small-M AB-swap tile configs.'
    )
    parser.add_argument(
        '--m-values',
        type=parse_int_list,
        default=DEFAULT_DECODE_M_VALUES,
        help='comma-separated public logical M values; default: 1,2,4',
    )
    parser.add_argument(
        '--quant',
        default='fp8',
        help='quant config: fp8, fp4xfp4, all, or comma-separated names',
    )
    parser.add_argument(
        '--shape-label',
        action='append',
        help='run only one named shape; repeat for multiple labels',
    )
    parser.add_argument(
        '--block-m-values',
        type=parse_int_list,
        default=(64, 96, 128),
        help='comma-separated runtime BLOCK_M values',
    )
    parser.add_argument(
        '--block-n-values',
        type=parse_int_list,
        default=(16, 32),
        help='comma-separated runtime BLOCK_N values; small-M swap usually uses 16 or 32',
    )
    parser.add_argument(
        '--block-k-values',
        type=parse_int_list,
        default=(64, 128),
        help='comma-separated runtime BLOCK_K values',
    )
    parser.add_argument(
        '--num-tests',
        type=int,
        default=10,
        help='profiler repetitions per tile candidate',
    )
    parser.add_argument(
        '--flush-l2',
        action='store_true',
        help='enable L2 flush in bench_kineto; default is hot-cache/no-flush',
    )
    parser.add_argument(
        '--skip-correctness',
        action='store_true',
        help='skip reference diff calculation',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed for each logical shape case',
    )
    parser.add_argument(
        '--out',
        default='auto',
        help='CSV output path; use auto for .omc/perf/dsv4_dense_normal_<timestamp>.csv',
    )
    return parser.parse_args()


def run_tile_case(
    tensors,
    quant_name: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_tests: int,
    flush_l2: bool,
    skip_correctness: bool,
) -> Dict[str, object]:
    gpu = load_gpu_modules()
    deep_gemm = gpu['deep_gemm']
    bench_kineto = gpu['bench_kineto']
    calc_diff = gpu['calc_diff']

    a, b, d, ref_d, recipe, recipe_a, recipe_b, max_diff = tensors

    def gemm():
        deep_gemm.sm120_fp8_gemm_bench(
            a,
            b,
            d,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            swap_ab=True,
        )

    gemm()
    gpu['torch'].cuda.synchronize()

    diff = None
    status = 'ok'
    if not skip_correctness:
        diff = float(calc_diff(d, ref_d))
        if diff >= max_diff:
            status = 'diff_fail'

    elapsed_s = bench_kineto(
        gemm,
        'gemm_',
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    return {
        'status': status,
        'time_us': elapsed_s * 1e6,
        'diff': diff,
        'max_diff': max_diff,
    }


def make_tensors(m: int, shape, quant_name: str, seed: int):
    gpu = load_gpu_modules()
    torch = gpu['torch']
    generate_normal = gpu['generate_normal']
    KernelType = gpu['KernelType']
    MajorTypeAB = gpu['MajorTypeAB']
    get_ue8m0_usage = gpu['get_ue8m0_usage']
    reset_seed = gpu['reset_seed']

    qc = quant_config_from_name(quant_name)
    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    recipe, recipe_a, recipe_b = qc.get_recipes()

    reset_seed(seed)
    a, b, c, d, ref_d = generate_normal(
        m,
        shape.n,
        shape.k,
        MajorTypeAB.KMajor,
        MajorTypeAB.KMajor,
        accumulate=False,
        out_dtype=torch.bfloat16,
        kernel_type=kernel_type,
        use_ue8m0=use_ue8m0,
        quant_config=qc,
    )
    return a, b, d, ref_d, recipe, recipe_a, recipe_b, qc.max_diff()


def main() -> int:
    args = parse_args()
    gpu = load_gpu_modules()
    get_arch_major = gpu['get_arch_major']
    assert get_arch_major() == 12, f'expected SM12, got arch_major={get_arch_major()}'

    quant_names = quant_names_from_arg(args.quant)
    valid_quant_names = {'fp8', 'fp4xfp4'}
    invalid = set(quant_names).difference(valid_quant_names)
    if invalid:
        raise ValueError(f'tile sweep supports non-mixed small-M swap only; invalid={sorted(invalid)}')

    shapes = selected_shapes(args.shape_label)
    output_path = make_output_path(args.out)
    rows: List[Dict[str, object]] = []

    print('DSv4 dense Normal small-M AB-swap tile sweep')
    print(f'M values: {",".join(str(m) for m in args.m_values)}')
    print(f'quant: {",".join(quant_names)}')
    print(f'BM: {args.block_m_values}, BN: {args.block_n_values}, BK: {args.block_k_values}')
    print(f'num_tests: {args.num_tests}, flush_l2: {args.flush_l2}')
    print()
    print(
        f'{"status":>9s} {"quant":>8s} {"logM":>5s} {"logN":>7s} {"K":>7s} '
        f'{"rtM":>7s} {"rtN":>5s} {"BM":>4s} {"BN":>4s} {"BK":>4s} '
        f'{"time_us":>10s} {"diff":>12s} label'
    )
    print('-' * 112)

    for quant_name in quant_names:
        for m in args.m_values:
            for shape in shapes:
                tensors = make_tensors(m, shape, quant_name, args.seed)
                runtime_m, runtime_n, runtime_k, swap_ab = expected_runtime_shape(m, shape, quant_name)
                if not swap_ab:
                    raise ValueError(f'case is not eligible for small-M swap: M={m}, quant={quant_name}')
                for block_m in args.block_m_values:
                    for block_n in args.block_n_values:
                        for block_k in args.block_k_values:
                            base = {
                                'quant': quant_name,
                                'logical_m': m,
                                'logical_n': shape.n,
                                'logical_k': shape.k,
                                'runtime_m': runtime_m,
                                'runtime_n': runtime_n,
                                'runtime_k': runtime_k,
                                'block_m': block_m,
                                'block_n': block_n,
                                'block_k': block_k,
                                'label': shape.label,
                                'num_tests': args.num_tests,
                                'flush_l2': args.flush_l2,
                                'note': shape.note,
                            }
                            try:
                                result = run_tile_case(
                                    tensors,
                                    quant_name,
                                    block_m,
                                    block_n,
                                    block_k,
                                    args.num_tests,
                                    args.flush_l2,
                                    args.skip_correctness,
                                )
                                row = {**base, **result}
                                diff = 'skip' if row['diff'] is None else f'{row["diff"]:.3e}'
                                print(
                                    f'{row["status"]:>9s} {quant_name:>8s} {m:5d} {shape.n:7d} {shape.k:7d} '
                                    f'{runtime_m:7d} {runtime_n:5d} {block_m:4d} {block_n:4d} {block_k:4d} '
                                    f'{row["time_us"]:10.2f} {diff:>12s} {shape.label}'
                                )
                            except Exception as exc:
                                row = {**base, 'status': 'error', 'error': repr(exc)}
                                print(
                                    f'{"error":>9s} {quant_name:>8s} {m:5d} {shape.n:7d} {shape.k:7d} '
                                    f'{runtime_m:7d} {runtime_n:5d} {block_m:4d} {block_n:4d} {block_k:4d} '
                                    f'{"-":>10s} {"-":>12s} {shape.label}: {exc}'
                                )
                            rows.append(row)

    if output_path is not None:
        fieldnames = [
            'status', 'quant', 'logical_m', 'logical_n', 'logical_k',
            'runtime_m', 'runtime_n', 'runtime_k', 'block_m', 'block_n',
            'block_k', 'label', 'time_us', 'diff', 'max_diff', 'num_tests',
            'flush_l2', 'error', 'note',
        ]
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f'wrote {output_path}')

    best_by_case = defaultdict(list)
    for row in rows:
        if row.get('status') != 'ok':
            continue
        key = (row['quant'], row['logical_m'], row['logical_n'], row['logical_k'])
        best_by_case[key].append(row)

    if best_by_case:
        print()
        print('best per logical case:')
        for key, candidates in sorted(best_by_case.items()):
            best = min(candidates, key=lambda row: row['time_us'])
            print(
                f'  quant={key[0]} M={key[1]} N={key[2]} K={key[3]}: '
                f'BM={best["block_m"]} BN={best["block_n"]} BK={best["block_k"]}, '
                f'{best["time_us"]:.2f} us'
            )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
