"""DSv4 dense Normal GEMM benchmark for SM120/SM121.

This benchmark intentionally excludes Grouped GEMM, BMM, and einsum paths.
It exercises only the public Normal dense API:

    deep_gemm.fp8_fp4_gemm_nt(a, b, d, ...)

The default quantization is FP8 x FP8 because the hot Normal dense buckets in
the reference decode trace have is_fp4_a=false and is_fp4_b=false.
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(__file__)
TESTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, REPO_ROOT)

_GPU_MODULES = None


def load_gpu_modules():
    global _GPU_MODULES
    if _GPU_MODULES is None:
        import torch
        import deep_gemm
        from deep_gemm.testing import bench_kineto, calc_diff, count_bytes, get_arch_major
        from generators import (
            generate_normal, KernelType, MajorTypeAB, QuantConfig,
            get_ue8m0_usage, reset_seed
        )
        _GPU_MODULES = {
            'torch': torch,
            'deep_gemm': deep_gemm,
            'bench_kineto': bench_kineto,
            'calc_diff': calc_diff,
            'count_bytes': count_bytes,
            'get_arch_major': get_arch_major,
            'generate_normal': generate_normal,
            'KernelType': KernelType,
            'MajorTypeAB': MajorTypeAB,
            'QuantConfig': QuantConfig,
            'get_ue8m0_usage': get_ue8m0_usage,
            'reset_seed': reset_seed,
        }
    return _GPU_MODULES


@dataclass(frozen=True)
class DenseNkShape:
    label: str
    n: int
    k: int
    note: str


DENSE_NK_SHAPES: Tuple[DenseNkShape, ...] = (
    DenseNkShape(
        label='logical_n4096_k1024',
        n=4096,
        k=1024,
        note='Small-M AB-swap runtime bucket <0,M,1024>; hot in the decode trace',
    ),
    DenseNkShape(
        label='logical_n4096_k4096',
        n=4096,
        k=4096,
        note='Small-M AB-swap runtime bucket <0,M,4096>; hot in the decode trace',
    ),
    DenseNkShape(
        label='logical_n1024_k4096',
        n=1024,
        k=4096,
        note='Small-M dense projection candidate with runtime K=4096',
    ),
    DenseNkShape(
        label='logical_n2048_k4096',
        n=2048,
        k=4096,
        note='Small-M dense projection candidate with runtime K=4096',
    ),
    DenseNkShape(
        label='logical_n8192_k1024',
        n=8192,
        k=1024,
        note='Large-N prefill dense candidate with runtime K=1024',
    ),
    DenseNkShape(
        label='logical_n16384_k1024',
        n=16384,
        k=1024,
        note='Large-N dense candidate with runtime K=1024',
    ),
)

DEFAULT_DECODE_M_VALUES: Tuple[int, ...] = (1, 2, 4)
DEFAULT_PREFILL_M_VALUES: Tuple[int, ...] = (64, 96, 128)


def parse_int_list(value: str) -> Tuple[int, ...]:
    items = [item.strip() for item in value.split(',') if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError('expected a comma-separated integer list')
    try:
        return tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def quant_config_from_name(name: str):
    QuantConfig = load_gpu_modules()['QuantConfig']
    configs = {
        'fp8': QuantConfig(),
        'fp8xfp4': QuantConfig((128, 32, False, True)),
        'fp4xfp4': QuantConfig((32, 32, True, True)),
    }
    return configs[name]


def quant_names_from_arg(value: str) -> Tuple[str, ...]:
    if value == 'all':
        return ('fp8', 'fp8xfp4', 'fp4xfp4')
    return tuple(part.strip() for part in value.split(',') if part.strip())


def selected_shapes(labels: Optional[Sequence[str]]) -> Tuple[DenseNkShape, ...]:
    if labels is None:
        return DENSE_NK_SHAPES

    label_set = set(labels)
    shapes = tuple(shape for shape in DENSE_NK_SHAPES if shape.label in label_set)
    missing = label_set.difference(shape.label for shape in shapes)
    if missing:
        available = ', '.join(shape.label for shape in DENSE_NK_SHAPES)
        raise ValueError(f'unknown shape labels {sorted(missing)}; available labels: {available}')
    return shapes


def make_output_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if path == 'auto':
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join('.omc', 'perf', f'dsv4_dense_normal_{stamp}.csv')
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def expected_runtime_shape(logical_m: int, shape: DenseNkShape, quant_name: str) -> Tuple[int, int, int, bool]:
    # SM120 plain dense small-M path swaps A/B for non-mixed FP8/FP4 cases.
    # The public API remains [M,K] @ [N,K].T, but the runtime descriptor sees
    # m=N, n=M, k=K when this path is eligible.
    swap_ab = logical_m <= 16 and quant_name in ('fp8', 'fp4xfp4')
    if swap_ab:
        return shape.n, logical_m, shape.k, True
    return logical_m, shape.n, shape.k, False


def run_case(
    m: int,
    shape: DenseNkShape,
    quant_name: str,
    num_tests: int,
    flush_l2: bool,
    skip_correctness: bool,
    seed: int,
) -> Dict[str, object]:
    gpu = load_gpu_modules()
    torch = gpu['torch']
    deep_gemm = gpu['deep_gemm']
    bench_kineto = gpu['bench_kineto']
    calc_diff = gpu['calc_diff']
    count_bytes = gpu['count_bytes']
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

    def gemm():
        deep_gemm.fp8_fp4_gemm_nt(
            a, b, d, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b
        )

    gemm()
    torch.cuda.synchronize()

    diff = None
    status = 'ok'
    if not skip_correctness:
        diff = float(calc_diff(d, ref_d))
        if diff >= qc.max_diff():
            status = 'diff_fail'

    elapsed_s = bench_kineto(
        gemm,
        'gemm_',
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )

    flops = 2 * m * shape.n * shape.k
    bytes_moved = count_bytes(a, b, d)
    runtime_m, runtime_n, runtime_k, swap_ab = expected_runtime_shape(m, shape, quant_name)
    return {
        'status': status,
        'quant': quant_name,
        'm': m,
        'n': shape.n,
        'k': shape.k,
        'logical_m': m,
        'logical_n': shape.n,
        'logical_k': shape.k,
        'runtime_m': runtime_m,
        'runtime_n': runtime_n,
        'runtime_k': runtime_k,
        'swap_ab': swap_ab,
        'label': shape.label,
        'time_us': elapsed_s * 1e6,
        'tflops': flops / elapsed_s / 1e12 if elapsed_s else 0.0,
        'bandwidth_gbs': bytes_moved / elapsed_s / 1e9 if elapsed_s else 0.0,
        'diff': diff,
        'max_diff': qc.max_diff(),
        'num_tests': num_tests,
        'flush_l2': flush_l2,
        'note': shape.note,
    }


def print_shape_catalog() -> None:
    print('available dense NK shapes:')
    for shape in DENSE_NK_SHAPES:
        runtime_m, runtime_n, runtime_k, swap_ab = expected_runtime_shape(1, shape, 'fp8')
        print(
            f'  {shape.label}: logical N={shape.n}, K={shape.k}; '
            f'M=1 runtime=<{runtime_m},{runtime_n},{runtime_k}>, swap_ab={swap_ab} '
            f'({shape.note})'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark DSv4 dense Normal GEMM shapes on SM120/SM121.'
    )
    parser.add_argument(
        '--m-values',
        type=parse_int_list,
        default=DEFAULT_DECODE_M_VALUES,
        help='comma-separated M values; default: 1,2,4',
    )
    parser.add_argument(
        '--include-prefill-m',
        action='store_true',
        help='append 64,96,128 to --m-values',
    )
    parser.add_argument(
        '--quant',
        default='fp8',
        help='quant config: fp8, fp8xfp4, fp4xfp4, all, or comma-separated names',
    )
    parser.add_argument(
        '--shape-label',
        action='append',
        help='run only one named shape; repeat for multiple labels',
    )
    parser.add_argument(
        '--list-shapes',
        action='store_true',
        help='print available shape labels and exit',
    )
    parser.add_argument(
        '--num-tests',
        type=int,
        default=30,
        help='number of profiler repetitions per case',
    )
    parser.add_argument(
        '--no-flush-l2',
        action='store_true',
        help='disable the default L2 flush in bench_kineto',
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
        help='random seed for each case',
    )
    parser.add_argument(
        '--out',
        default=None,
        help='CSV output path; use "auto" for .omc/perf/dsv4_dense_normal_<timestamp>.csv',
    )
    parser.add_argument(
        '--allow-non-sm12',
        action='store_true',
        help='allow running outside SM12 for local smoke testing',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_shapes:
        print_shape_catalog()
        return 0

    if not args.allow_non_sm12:
        get_arch_major = load_gpu_modules()['get_arch_major']
        assert get_arch_major() == 12, f'expected SM12, got arch_major={get_arch_major()}'

    quant_names = quant_names_from_arg(args.quant)
    valid_quant_names = {'fp8', 'fp8xfp4', 'fp4xfp4'}
    invalid_quant_names = set(quant_names).difference(valid_quant_names)
    if invalid_quant_names:
        raise ValueError(f'unknown quant configs: {sorted(invalid_quant_names)}')
    if not quant_names:
        raise ValueError('at least one quant config is required')

    m_values = list(args.m_values)
    if args.include_prefill_m:
        m_values.extend(DEFAULT_PREFILL_M_VALUES)

    shapes = selected_shapes(args.shape_label)
    output_path = make_output_path(args.out)

    rows: List[Dict[str, object]] = []
    print('DSv4 dense Normal GEMM benchmark')
    print(f'M values: {",".join(str(m) for m in m_values)}')
    print(f'quant: {",".join(quant_names)}')
    print(f'num_tests: {args.num_tests}, flush_l2: {not args.no_flush_l2}')
    print()
    print(
        f'{"status":>9s} {"quant":>8s} {"logM":>5s} {"logN":>7s} {"K":>7s} '
        f'{"rtM":>7s} {"rtN":>5s} {"swap":>5s} {"time_us":>10s} '
        f'{"TFLOPS":>9s} {"GB/s":>9s} {"diff":>12s} label'
    )
    print('-' * 123)

    for quant_name in quant_names:
        for m in m_values:
            for shape in shapes:
                try:
                    row = run_case(
                        m=m,
                        shape=shape,
                        quant_name=quant_name,
                        num_tests=args.num_tests,
                        flush_l2=not args.no_flush_l2,
                        skip_correctness=args.skip_correctness,
                        seed=args.seed,
                    )
                    rows.append(row)
                    diff = 'skip' if row['diff'] is None else f'{row["diff"]:.3e}'
                    print(
                        f'{row["status"]:>9s} {row["quant"]:>8s} {row["logical_m"]:5d} '
                        f'{row["logical_n"]:7d} {row["logical_k"]:7d} '
                        f'{row["runtime_m"]:7d} {row["runtime_n"]:5d} '
                        f'{str(row["swap_ab"]):>5s} {row["time_us"]:10.2f} '
                        f'{row["tflops"]:9.2f} {row["bandwidth_gbs"]:9.1f} '
                        f'{diff:>12s} {row["label"]}'
                    )
                except Exception as exc:
                    row = {
                        'status': 'error',
                        'quant': quant_name,
                        'm': m,
                        'n': shape.n,
                        'k': shape.k,
                        'label': shape.label,
                        'error': repr(exc),
                        'note': shape.note,
                    }
                    rows.append(row)
                    runtime_m, runtime_n, _, swap_ab = expected_runtime_shape(m, shape, quant_name)
                    print(
                        f'{"error":>9s} {quant_name:>8s} {m:5d} {shape.n:7d} '
                        f'{shape.k:7d} {runtime_m:7d} {runtime_n:5d} '
                        f'{str(swap_ab):>5s} {"-":>10s} {"-":>9s} {"-":>9s} '
                        f'{"-":>12s} {shape.label}: {exc}'
                    )

    if output_path is not None:
        fieldnames = [
            'status', 'quant', 'm', 'n', 'k', 'logical_m', 'logical_n',
            'logical_k', 'runtime_m', 'runtime_n', 'runtime_k', 'swap_ab',
            'label', 'time_us', 'tflops', 'bandwidth_gbs', 'diff',
            'max_diff', 'num_tests', 'flush_l2', 'error', 'note',
        ]
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f'wrote {output_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
