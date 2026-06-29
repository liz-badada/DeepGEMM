from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import statistics
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
RESULTS_TSV = REPO_ROOT / "avo" / "g2_masked_results.tsv"
RESULTS_JSONL = REPO_ROOT / "avo" / "g2_masked_results.jsonl"

NUM_GROUPS = 6
MAX_M = 4096
EXPECTED_M = 1024
N = 4096
K = 4096
SEED = 20260525
COMPILED_DIMS = os.environ.get("DG_COMPILED_DIMS", "nk")
G2_SOURCE_MASKED_M = [744, 940, 722, 747, 1157, 1082]
G2_FIXED_MASKED_M = [768, 896, 768, 768, 1152, 1024]
G2_MASK_POLICY = "fixed_round_nearest_128_from_pr324_g2"

TSV_COLS = [
    "timestamp", "iter", "stage", "variant",
    "groups", "max_m", "expected_m", "valid_m", "masked_m",
    "n", "k", "layout", "out_dtype", "dtype",
    "median_us", "tflops", "gb_s", "diff_vs_ref", "num_sms", "notes",
]


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _ensure_tsv_header() -> None:
    if RESULTS_TSV.exists() and RESULTS_TSV.stat().st_size > 0:
        return
    RESULTS_TSV.write_text("\t".join(TSV_COLS) + "\n")


def _record(stage: str, iteration: int, variant: str, **kw) -> None:
    row = {"timestamp": _now(), "iter": iteration, "stage": stage, "variant": variant}
    row.update(kw)
    _ensure_tsv_header()
    with RESULTS_TSV.open("a") as f:
        f.write("\t".join(str(row.get(c, "")) for c in TSV_COLS) + "\n")
    with RESULTS_JSONL.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _seed(seed: int = SEED) -> None:
    import torch  # type: ignore

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _num_sms() -> int:
    try:
        import deep_gemm  # type: ignore

        return int(deep_gemm.get_num_sms())
    except Exception:
        return -1


def _build_inputs():
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))

    import deep_gemm  # type: ignore
    import torch  # type: ignore
    from generators import (  # type: ignore
        KernelType,
        QuantConfig,
        generate_m_grouped_masked,
        get_ue8m0_usage,
    )

    _seed()
    alignment = int(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(EXPECTED_M * 1.2)))
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    kernel_type = KernelType.Kernel1D1D
    quant_config = QuantConfig((32, 32, True, True))
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    a, b, masked_m, _psum_m, d, ref_d = generate_m_grouped_masked(
        NUM_GROUPS, MAX_M, EXPECTED_M, N, K,
        use_ue8m0=use_ue8m0,
        use_psum_layout=False,
        quant_config=quant_config,
    )
    torch.cuda.synchronize()
    return a, b, masked_m, d, ref_d, recipe, recipe_a, recipe_b, use_ue8m0, quant_config, alignment


def _apply_g2_mask(masked_m) -> list[int]:
    # G2 benchmark policy: use the PR324 generated mask rounded to nearest 128.
    masked_m.copy_(masked_m.new_tensor(G2_FIXED_MASKED_M))
    return list(G2_FIXED_MASKED_M)


def _run(a, b, d, masked_m, recipe, recipe_a, recipe_b, use_ue8m0) -> None:
    import deep_gemm  # type: ignore

    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
        a, b, d, masked_m, int(EXPECTED_M * 1.2),
        disable_ue8m0_cast=not use_ue8m0,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        compiled_dims=COMPILED_DIMS,
    )


def _check_diff(d, ref_d, masked_m, quant_config) -> float:
    from deep_gemm.testing import calc_diff  # type: ignore

    max_diff = 0.0
    for group_idx in range(NUM_GROUPS):
        m = int(masked_m[group_idx].item())
        if m == 0:
            continue
        max_diff = max(max_diff, float(calc_diff(d[group_idx, :m], ref_d[group_idx, :m])))
    if max_diff >= quant_config.max_diff():
        raise RuntimeError(f"correctness diff too high: {max_diff:.6f} >= {quant_config.max_diff():.6f}")
    return max_diff


def stage_preflight(iteration: int, variant: str) -> None:
    import deep_gemm  # type: ignore
    import torch  # type: ignore
    from deep_gemm.testing import get_arch_major  # type: ignore

    notes = [
        f"torch={torch.__version__}",
        f"cuda_avail={torch.cuda.is_available()}",
        f"deep_gemm_path={deep_gemm.__path__[0]}",
        f"arch_major={get_arch_major()}",
    ]
    if torch.cuda.is_available():
        notes.append(f"gpu={torch.cuda.get_device_name(0)}")
        notes.append(f"cap={'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
    _record("preflight", iteration, variant, num_sms=_num_sms(), notes=";".join(notes))


def stage_tiny(iteration: int, variant: str) -> None:
    import torch  # type: ignore

    a, b, masked_m, d, ref_d, recipe, recipe_a, recipe_b, use_ue8m0, qc, alignment = _build_inputs()
    masked_m_values = _apply_g2_mask(masked_m)
    _run(a, b, d, masked_m, recipe, recipe_a, recipe_b, use_ue8m0)
    torch.cuda.synchronize()
    diff = _check_diff(d, ref_d, masked_m, qc)
    valid_m = int(masked_m.sum().item())
    _record(
        "tiny", iteration, variant,
        groups=NUM_GROUPS, max_m=MAX_M, expected_m=EXPECTED_M,
        valid_m=valid_m, masked_m=masked_m_values,
        mask_policy=G2_MASK_POLICY,
        n=N, k=K, layout="NN", out_dtype="bf16", dtype="fp4_x_fp4",
        diff_vs_ref=f"{diff:.5f}", num_sms=_num_sms(),
        notes=(
            f"dispatch_ok;alignment={alignment};recipe_a=(1,32);recipe_b=(1,32);"
            f"compiled_dims={COMPILED_DIMS};mask_policy={G2_MASK_POLICY};"
            f"source_masked_m={G2_SOURCE_MASKED_M}"
        ),
    )


def stage_bench(iteration: int, variant: str, num_repeats: int = 5) -> None:
    import torch  # type: ignore
    from deep_gemm.testing import bench_kineto, count_bytes  # type: ignore

    a, b, masked_m, d, ref_d, recipe, recipe_a, recipe_b, use_ue8m0, qc, alignment = _build_inputs()
    masked_m_values = _apply_g2_mask(masked_m)
    _run(a, b, d, masked_m, recipe, recipe_a, recipe_b, use_ue8m0)
    torch.cuda.synchronize()
    diff = _check_diff(d, ref_d, masked_m, qc)
    valid_m = int(masked_m.sum().item())

    repeats = [
        bench_kineto(lambda: _run(a, b, d, masked_m, recipe, recipe_a, recipe_b, use_ue8m0),
                     "gemm_", suppress_kineto_output=True)
        for _ in range(num_repeats)
    ]
    median_t = float(statistics.median(repeats))
    tflops = 2.0 * valid_m * N * K / median_t / 1e12
    # Match the upstream masked benchmark byte accounting.
    gb_s = (count_bytes(a, d) * valid_m / (MAX_M * NUM_GROUPS) + count_bytes(b)) / 1e9 / median_t
    _record(
        "bench", iteration, variant,
        groups=NUM_GROUPS, max_m=MAX_M, expected_m=EXPECTED_M,
        valid_m=valid_m, masked_m=masked_m_values,
        mask_policy=G2_MASK_POLICY,
        n=N, k=K, layout="NN", out_dtype="bf16", dtype="fp4_x_fp4",
        median_us=f"{median_t * 1e6:.1f}",
        tflops=f"{tflops:.1f}",
        gb_s=f"{gb_s:.1f}",
        diff_vs_ref=f"{diff:.5f}",
        num_sms=_num_sms(),
        notes=(
            f"ok;repeats_us={[round(t * 1e6, 1) for t in repeats]};"
            f"min_us={min(repeats) * 1e6:.1f};max_us={max(repeats) * 1e6:.1f};"
            f"alignment={alignment};seed={SEED};recipe_a=(1,32);recipe_b=(1,32);"
            f"compiled_dims={COMPILED_DIMS};mask_policy={G2_MASK_POLICY};"
            f"source_masked_m={G2_SOURCE_MASKED_M}"
        ),
    )


STAGES = {"preflight": stage_preflight, "tiny": stage_tiny, "bench": stage_bench}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--variant", type=str, default="g2_masked_fp4")
    parser.add_argument("--num-sms", type=int, default=None)
    args = parser.parse_args()
    try:
        if args.num_sms is not None:
            import deep_gemm  # type: ignore

            deep_gemm.set_num_sms(args.num_sms)
        STAGES[args.stage](args.iter, args.variant)
        return 0
    except Exception as exc:
        tb = traceback.format_exc().replace("\n", " | ").replace("\t", " ")
        if len(tb) > 800:
            tb = tb[:800] + "...<truncated>"
        _record(args.stage, args.iter, args.variant, num_sms=_num_sms(), notes=f"ERROR:{type(exc).__name__}:{tb}")
        print(f"[bench_g2_masked_fp4] ERROR in stage={args.stage}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
