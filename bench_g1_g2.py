from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

for include_dir in (
    REPO_ROOT / "third-party" / "cutlass" / "include",
    REPO_ROOT / "third-party" / "fmt" / "include",
    REPO_ROOT / "deep_gemm" / "include",
):
    if include_dir.exists():
        old_cpath = os.environ.get("CPATH", "")
        old_cplus = os.environ.get("CPLUS_INCLUDE_PATH", "")
        include = str(include_dir)
        if include not in old_cpath.split(":"):
            os.environ["CPATH"] = include if not old_cpath else f"{include}:{old_cpath}"
        if include not in old_cplus.split(":"):
            os.environ["CPLUS_INCLUDE_PATH"] = include if not old_cplus else f"{include}:{old_cplus}"

GB10_FP4_DENSE_TFLOPS = 500.0
GB10_MEMORY_BANDWIDTH_GBPS = 273.0
RTX_PRO_5000_FP4_DENSE_TFLOPS = 1032.0
RTX_PRO_5000_MEMORY_BANDWIDTH_GBPS = 1344.0

G1 = {
    "groups": 4,
    "expected_m": 8192,
    "target_m": 36096,
    "n": 6144,
    "k": 7168,
    "seed": 124,
    "native_repeats": 3,
}

G2 = {
    "groups": 6,
    "max_m": 4096,
    "expected_m": 1024,
    "source_masked_m": [744, 940, 722, 747, 1157, 1082],
    "fixed_masked_m": [768, 896, 768, 768, 1152, 1024],
    "mask_policy": "fixed_round_nearest_128_from_pr324_g2",
    "n": 4096,
    "k": 4096,
    "seed": 20260525,
    "compiled_dims": "nk",
    "native_repeats": 5,
}


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def detect_hardware_profile() -> str:
    import deep_gemm
    import torch

    device_name = torch.cuda.get_device_name(0).lower()
    num_sms = int(deep_gemm.get_num_sms())
    if "gb10" in device_name or num_sms == 48:
        return "dgx_spark"
    if "5000" in device_name or num_sms == 110:
        return "rtx_pro_5000"
    raise RuntimeError(
        f"Cannot infer hardware profile from device={torch.cuda.get_device_name(0)!r}, num_sms={num_sms}. "
        "Pass --hardware dgx_spark or --hardware rtx_pro_5000."
    )


def hardware_config(profile: str) -> dict:
    if profile == "dgx_spark":
        return {
            "profile": profile,
            "compute_peak_tflops": GB10_FP4_DENSE_TFLOPS,
            "memory_peak_gbps": GB10_MEMORY_BANDWIDTH_GBPS,
            "g1_metric": "compute",
            "g2_metric": "memory",
        }
    if profile == "rtx_pro_5000":
        return {
            "profile": profile,
            "compute_peak_tflops": RTX_PRO_5000_FP4_DENSE_TFLOPS,
            "memory_peak_gbps": RTX_PRO_5000_MEMORY_BANDWIDTH_GBPS,
            "g1_metric": "compute",
            "g2_metric": "compute",
        }
    raise ValueError(profile)


def check_arch() -> None:
    from deep_gemm.testing import get_arch_major

    arch_major = get_arch_major()
    if arch_major != 12:
        raise RuntimeError(f"Expected SM120/SM121 arch_major=12, got {arch_major}")


def bench_g1() -> dict:
    import deep_gemm
    import torch
    from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
    from deep_gemm.utils import align, get_mk_alignment_for_contiguous_layout
    from generators import (
        KernelType,
        MajorTypeAB,
        QuantConfig,
        generate_m_grouped_contiguous,
        get_ue8m0_usage,
    )

    seed_all(G1["seed"])
    alignment = int(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    quant_config = QuantConfig((32, 32, True, True))
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
        G1["groups"],
        G1["expected_m"],
        G1["n"],
        G1["k"],
        MajorTypeAB.KMajor,
        MajorTypeAB.MNMajor,
        use_ue8m0=use_ue8m0,
        use_psum_layout=True,
        quant_config=quant_config,
    )
    if int(m) != G1["target_m"]:
        raise RuntimeError(f"wrong G1 generated M: got {int(m)}, expected {G1['target_m']}")

    def run() -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a,
            b,
            d,
            grouped_layout,
            disable_ue8m0_cast=not use_ue8m0,
            use_psum_layout=True,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

    run()
    torch.cuda.synchronize()

    max_diff = 0.0
    for group_idx in range(G1["groups"]):
        prev = 0 if group_idx == 0 else grouped_layout[group_idx - 1]
        prev_end = 0 if group_idx == 0 else int(prev.item() if hasattr(prev, "item") else prev)
        start = 0 if group_idx == 0 else align(prev_end, get_mk_alignment_for_contiguous_layout())
        current = grouped_layout[group_idx]
        end = int(current.item() if hasattr(current, "item") else current)
        max_diff = max(max_diff, float(calc_diff(d[start:end], ref_d[start:end])))
    if max_diff >= quant_config.max_diff():
        raise RuntimeError(f"G1 correctness diff too high: {max_diff:.6f} >= {quant_config.max_diff():.6f}")

    repeats = [
        bench_kineto(lambda: run(), "gemm_", suppress_kineto_output=True)
        for _ in range(G1["native_repeats"])
    ]
    latency_s = float(statistics.median(repeats))
    flops_total = float(2.0 * int(m) * G1["n"] * G1["k"])
    tensor_bytes_total = float(count_bytes(a, b, d))

    return {
        "case": "G1 contiguous / psum",
        "kernel": "m_grouped_fp8_fp4_gemm_nt_contiguous",
        "metric": "compute",
        "groups": G1["groups"],
        "m": int(m),
        "expected_m": G1["expected_m"],
        "n": G1["n"],
        "k": G1["k"],
        "latency_us": latency_s * 1e6,
        "tflops": flops_total / latency_s / 1e12,
        "gb_s": tensor_bytes_total / latency_s / 1e9,
        "flops_total": flops_total,
        "memory_bytes_total": tensor_bytes_total,
        "max_diff": float(max_diff),
        "native_repeats_us": [float(t * 1e6) for t in repeats],
        "alignment": alignment,
        "recipe_a": [1, 32],
        "recipe_b": [1, 32],
        "num_sms": int(deep_gemm.get_num_sms()),
    }


def bench_g2() -> dict:
    import deep_gemm
    import torch
    from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
    from generators import KernelType, QuantConfig, generate_m_grouped_masked, get_ue8m0_usage

    seed_all(G2["seed"])
    alignment = int(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(G2["expected_m"] * 1.2)))
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    quant_config = QuantConfig((32, 32, True, True))
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    a, b, masked_m, _psum_m, d, ref_d = generate_m_grouped_masked(
        G2["groups"],
        G2["max_m"],
        G2["expected_m"],
        G2["n"],
        G2["k"],
        use_ue8m0=use_ue8m0,
        use_psum_layout=False,
        quant_config=quant_config,
    )
    masked_m.copy_(masked_m.new_tensor(G2["fixed_masked_m"]))

    def run() -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
            a,
            b,
            d,
            masked_m,
            int(G2["expected_m"] * 1.2),
            disable_ue8m0_cast=not use_ue8m0,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            compiled_dims=G2["compiled_dims"],
        )

    run()
    torch.cuda.synchronize()

    max_diff = 0.0
    for group_idx in range(G2["groups"]):
        m_i = int(masked_m[group_idx].item())
        if m_i:
            max_diff = max(max_diff, float(calc_diff(d[group_idx, :m_i], ref_d[group_idx, :m_i])))
    if max_diff >= quant_config.max_diff():
        raise RuntimeError(f"G2 correctness diff too high: {max_diff:.6f} >= {quant_config.max_diff():.6f}")

    repeats = [
        bench_kineto(lambda: run(), "gemm_", suppress_kineto_output=True)
        for _ in range(G2["native_repeats"])
    ]
    latency_s = float(statistics.median(repeats))
    valid_m = int(masked_m.sum().item())
    valid_scale = float(valid_m) / float(G2["max_m"] * G2["groups"])

    a_payload_bytes = float(count_bytes(a[0]))
    a_scale_bytes = float(count_bytes(a[1]))
    b_payload_bytes = float(count_bytes(b[0]))
    b_scale_bytes = float(count_bytes(b[1]))
    a_bytes_total = float(count_bytes(a))
    b_bytes_total = float(count_bytes(b))
    d_bytes_total = float(count_bytes(d))
    read_bytes = float(a_bytes_total * valid_scale + b_bytes_total)
    write_bytes = float(d_bytes_total * valid_scale)
    memory_bytes_total = read_bytes + write_bytes
    flops_total = float(2.0 * valid_m * G2["n"] * G2["k"])

    return {
        "case": "G2 masked round128",
        "kernel": "m_grouped_fp8_fp4_gemm_nt_masked",
        "metric": "auto",
        "groups": G2["groups"],
        "max_m": G2["max_m"],
        "expected_m": G2["expected_m"],
        "valid_m": valid_m,
        "masked_m": [int(x) for x in masked_m.cpu().tolist()],
        "source_masked_m": G2["source_masked_m"],
        "mask_policy": G2["mask_policy"],
        "n": G2["n"],
        "k": G2["k"],
        "latency_us": latency_s * 1e6,
        "tflops": flops_total / latency_s / 1e12,
        "gb_s": memory_bytes_total / latency_s / 1e9,
        "flops_total": flops_total,
        "memory_bytes_total": memory_bytes_total,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "valid_m_scale": valid_scale,
        "tensor_bytes": {
            "a": {"payload": a_payload_bytes, "scale": a_scale_bytes, "total": a_bytes_total},
            "b": {"payload": b_payload_bytes, "scale": b_scale_bytes, "total": b_bytes_total},
            "d": {"total": d_bytes_total},
        },
        "max_diff": float(max_diff),
        "native_repeats_us": [float(t * 1e6) for t in repeats],
        "alignment": alignment,
        "compiled_dims": G2["compiled_dims"],
        "recipe_a": [1, 32],
        "recipe_b": [1, 32],
        "num_sms": int(deep_gemm.get_num_sms()),
    }


def add_sol(row: dict, hw: dict, *, metric: str) -> dict:
    out = dict(row)
    out["metric"] = metric
    if metric == "compute":
        out["sol_pct"] = out["tflops"] / hw["compute_peak_tflops"] * 100.0
        out["sol_peak"] = hw["compute_peak_tflops"]
        out["sol_unit"] = "TFLOP/s"
    elif metric == "memory":
        out["sol_pct"] = out["gb_s"] / hw["memory_peak_gbps"] * 100.0
        out["sol_peak"] = hw["memory_peak_gbps"]
        out["sol_unit"] = "GB/s"
    else:
        raise ValueError(metric)
    return out


def print_table(rows: list[dict], hw: dict) -> None:
    print("Bench G1 G2")
    print(f"Hardware profile: {hw['profile']}")
    print("Timing: deep_gemm.testing.bench_kineto, native median repeats G1=3 G2=5")
    print()
    print(
        f"{'Case':<24} {'Metric':<8} {'Latency(us)':>12} {'TFLOPS':>10} "
        f"{'GB/s':>10} {'SOL%':>8} {'diff':>10}"
    )
    print("-" * 90)
    for row in rows:
        print(
            f"{row['case']:<24} {row['metric']:<8} {row['latency_us']:12.1f} "
            f"{row['tflops']:10.1f} {row['gb_s']:10.1f} {row['sol_pct']:8.1f} "
            f"{row['max_diff']:10.6f}"
        )
    print()
    for row in rows:
        if row["metric"] == "compute":
            formula = "SOL = TFLOPS / peak_TFLOPS * 100"
        else:
            formula = "SOL = GB/s / peak_GB/s * 100"
        print(f"{row['case']}: {formula}; peak={row['sol_peak']} {row['sol_unit']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DeepGEMM G1 and G2 shapes.")
    parser.add_argument("--case", choices=("all", "g1", "g2"), default="all")
    parser.add_argument("--hardware", choices=("auto", "dgx_spark", "rtx_pro_5000"), default="auto")
    parser.add_argument("--g2-metric", choices=("auto", "compute", "memory"), default="auto")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a table.")
    args = parser.parse_args()

    check_arch()
    profile = detect_hardware_profile() if args.hardware == "auto" else args.hardware
    hw = hardware_config(profile)
    g2_metric = hw["g2_metric"] if args.g2_metric == "auto" else args.g2_metric

    rows = []
    if args.case in ("all", "g1"):
        rows.append(add_sol(bench_g1(), hw, metric=hw["g1_metric"]))
    if args.case in ("all", "g2"):
        rows.append(add_sol(bench_g2(), hw, metric=g2_metric))

    if args.json:
        print(json.dumps({"hardware": hw, "results": rows}, indent=2, sort_keys=True))
    else:
        print_table(rows, hw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
