#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


GB10_FP4_DENSE_TFLOPS = 500.0
BENCH_KINETO_NUM_TESTS = 30
FULL_GATE_CONTIGUOUS_TOTAL = 18
FULL_GATE_MASKED_TOTAL = 15

KERNELS = {
    "g1": {
        "bench_script": "bench_mgroup_moe.py",
        "cmd": lambda py, index, variant: [
            py,
            "avo/bench_mgroup_moe.py",
            "mgroup_bench",
            "--iter",
            str(index),
            "--variant",
            variant,
        ],
        "jsonl": "avo/results.jsonl",
        "stage": "mgroup_bench",
        "kernel_name": "m_grouped_fp8_fp4_gemm_nt_contiguous_g1_psum_true",
        "shape": "groups=4,m=36096,n=6144,k=7168,layout=NN,psum_layout=true,expected_m_per_group=8192,gran_k=32",
        "native_repeats": 3,
    },
    "g2": {
        "bench_script": "bench_g2_masked_fp4.py",
        "cmd": lambda py, index, variant: [
            py,
            "avo/bench_g2_masked_fp4.py",
            "bench",
            "--iter",
            str(index),
            "--variant",
            variant,
        ],
        "jsonl": "avo/g2_masked_results.jsonl",
        "stage": "bench",
        "kernel_name": "m_grouped_fp8_fp4_gemm_nt_masked_g2",
        "shape": (
            "groups=6,max_m=4096,expected_m=1024,n=4096,k=4096,layout=NN,"
            "compiled_dims=nk,gran_k=32,masked_m=[768,896,768,768,1152,1024],"
            "mask_policy=fixed_round_nearest_128_from_pr324_g2"
        ),
        "native_repeats": 5,
    },
}


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def check_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = run_cmd(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd={cmd}\n"
            f"cwd={cwd}\n"
            f"stdout={proc.stdout[-6000:]}\n"
            f"stderr={proc.stderr[-6000:]}"
        )
    return proc


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mad(values: list[float]) -> float:
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def read_text_or_unknown(path: Path) -> str:
    try:
        text = path.read_text().strip()
        return text if text else "unknown"
    except OSError:
        return "unknown"


def collect_gpu_info(python_exe: str, repo: Path) -> dict:
    gpu_query = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.gr",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_fields = [x.strip() for x in gpu_query.stdout.splitlines()[0].split(",")] if gpu_query.stdout.strip() else []
    torch_query = run_cmd([python_exe, "-c", "import torch; print(torch.version.cuda)"], cwd=repo)
    cuda_version = torch_query.stdout.splitlines()[0].strip() if torch_query.returncode == 0 and torch_query.stdout.splitlines() else "unknown"
    sm_count_query = run_cmd(
        [python_exe, "-c", "import deep_gemm; print(deep_gemm.get_num_sms())"],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo), "DG_JIT_CACHE_DIR": "/tmp/kbc_deepgemm_probe_cache"},
    )
    sm_count = int(sm_count_query.stdout.strip()) if sm_count_query.returncode == 0 and sm_count_query.stdout.strip().isdigit() else 0
    hostname = read_text_or_unknown(Path("/proc/sys/kernel/hostname"))
    return {
        "machine_model": f"DGX Spark / {hostname}",
        "gpu_model": gpu_fields[0] if len(gpu_fields) > 0 else "NVIDIA GB10",
        "sm_count": sm_count,
        "driver_version": gpu_fields[1] if len(gpu_fields) > 1 else "unknown",
        "cuda_version": cuda_version,
        "clock_state": (
            f"not_locked,pstate={gpu_fields[2] if len(gpu_fields) > 2 else 'unknown'},"
            f"sm_clock_mhz={gpu_fields[3] if len(gpu_fields) > 3 else 'unknown'},"
            f"gr_clock_mhz={gpu_fields[4] if len(gpu_fields) > 4 else 'unknown'}"
        ),
        "sol_denominator_metric": "compute",
        "sol_denominator_mode": "dense_tensorcore",
        "sol_denominator_unit": "TFLOP/s",
        "sol_denominator_registry_key": "nvidia_gb10_fp4_dense_tensorcore",
        "sol_denominator_value": GB10_FP4_DENSE_TFLOPS,
    }


def prepare_repo(args: argparse.Namespace, role: str, commit: str, stdout_parts: list[str], stderr_parts: list[str]) -> Path:
    repo = Path(args.work_root) / f"DeepGEMM_{role}"
    if repo.exists():
        shutil.rmtree(repo)
    check_cmd(["git", "config", "--global", "--add", "safe.directory", args.source_repo])
    check_cmd(["git", "config", "--global", "--add", "safe.directory", str(Path(args.source_repo) / ".git")])
    check_cmd(["git", "clone", "--no-local", args.source_repo, str(repo)])
    check_cmd(["git", "fetch", args.bundle, "refs/heads/codex/mgroup-fp4-g1-opt:refs/heads/kbc_candidate"], cwd=repo)
    check_cmd(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo)
    check_cmd(["git", "checkout", "--force", commit], cwd=repo)
    check_cmd(["git", "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive"], cwd=repo)

    include_root = repo / "deep_gemm" / "include"
    for dirname in ("cute", "cutlass", "fmt"):
        candidates = [
            repo / "third-party" / "cutlass" / "include" / dirname,
            repo / "third-party" / dirname / "include" / dirname,
        ]
        src = next((candidate for candidate in candidates if candidate.exists()), None)
        if src is None:
            continue
        dst = include_root / dirname
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.copytree(src, dst)

    avo_dir = repo / "avo"
    avo_dir.mkdir(parents=True, exist_ok=True)
    for kernel in KERNELS.values():
        shutil.copy2(Path(args.run_root) / "harness" / kernel["bench_script"], avo_dir / kernel["bench_script"])

    env = {
        **os.environ,
        "DG_USE_LOCAL_VERSION": "0",
        "MAX_JOBS": os.environ.get("MAX_JOBS", "8"),
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0a"),
        "CC": os.environ.get("CC", "gcc"),
        "CXX": os.environ.get("CXX", "g++"),
        "CUDAHOSTCXX": os.environ.get("CUDAHOSTCXX", "g++"),
    }
    proc = check_cmd([args.python, "setup.py", "build_ext", "--inplace"], cwd=repo, env=env)
    stdout_parts.append(f"BUILD {role} {commit}\n{proc.stdout}")
    stderr_parts.append(f"BUILD {role} {commit}\n{proc.stderr}")
    return repo


def parse_full_gate_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in ("Contiguous", "Masked"):
        match = re.search(rf"{key}:\s+(\d+)/(\d+)\s+passed", output)
        if match:
            counts[f"{key.lower()}_passed"] = int(match.group(1))
            counts[f"{key.lower()}_total"] = int(match.group(2))
    return counts


def run_full_accuracy_gate(args: argparse.Namespace, repo: Path, role: str) -> dict:
    cache_dir = Path(args.work_root) / f"jit_cache_full_accuracy_{role}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    runner = f"""
import importlib.util
import pathlib
import sys
import torch

repo = pathlib.Path({str(repo)!r})
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location(
    "test_m_grouped_fp8",
    repo / "tests" / "sm120" / "test_m_grouped_fp8.py",
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
torch.manual_seed(0)
contiguous_ok = mod.test_m_grouped_contiguous()
masked_ok = mod.test_m_grouped_masked()
if not (contiguous_ok and masked_ok):
    raise SystemExit(2)
"""
    env = {
        **os.environ,
        "PYTHONPATH": f"{repo}:{repo / 'tests'}",
        "DG_USE_LOCAL_VERSION": "0",
        "DG_JIT_CACHE_DIR": str(cache_dir),
    }
    proc = run_cmd([args.python, "-c", runner], cwd=repo, env=env)
    output = proc.stdout + proc.stderr
    counts = parse_full_gate_counts(output)
    contiguous_passed = counts.get("contiguous_passed", 0)
    contiguous_total = counts.get("contiguous_total", FULL_GATE_CONTIGUOUS_TOTAL)
    masked_passed = counts.get("masked_passed", 0)
    masked_total = counts.get("masked_total", FULL_GATE_MASKED_TOTAL)
    failed_cases = max(contiguous_total - contiguous_passed, 0) + max(masked_total - masked_passed, 0)
    pass_gate = (
        proc.returncode == 0
        and contiguous_passed == FULL_GATE_CONTIGUOUS_TOTAL
        and contiguous_total == FULL_GATE_CONTIGUOUS_TOTAL
        and masked_passed == FULL_GATE_MASKED_TOTAL
        and masked_total == FULL_GATE_MASKED_TOTAL
    )
    return {
        "role": role,
        "pass": pass_gate,
        "returncode": proc.returncode,
        "contiguous_passed": contiguous_passed,
        "contiguous_total": contiguous_total,
        "masked_passed": masked_passed,
        "masked_total": masked_total,
        "failed_cases": failed_cases if proc.returncode != 1 or counts else failed_cases + 1,
        "log": output,
    }


def parse_native_row(kernel_name: str, repo: Path, variant: str) -> dict:
    meta = KERNELS[kernel_name]
    rows = [json.loads(line) for line in (repo / meta["jsonl"]).read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("stage") == meta["stage"] and row.get("variant") == variant]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {kernel_name} row for {variant}, got {len(selected)}")
    row = selected[0]
    notes = str(row.get("notes", ""))
    passed = not notes.startswith("ERROR:") and row.get("diff_vs_ref") not in (None, "")
    if not passed:
        return {"passed": False, "error": notes, "native_row": row}
    return {
        "passed": True,
        "latency_ms": float(row["median_us"]) / 1000.0,
        "tflops": float(row["tflops"]),
        "sol_pct": float(row["tflops"]) / GB10_FP4_DENSE_TFLOPS * 100.0,
        "max_diff": float(row["diff_vs_ref"]),
        "tolerance": 0.02,
        "native_row": row,
    }


def run_kernel_sample(
    kernel_name: str,
    repo: Path,
    role: str,
    index: int,
    args: argparse.Namespace,
    stdout_parts: list[str],
    stderr_parts: list[str],
) -> dict:
    meta = KERNELS[kernel_name]
    for rel in ("avo/results.tsv", "avo/results.jsonl", "avo/g2_masked_results.tsv", "avo/g2_masked_results.jsonl"):
        path = repo / rel
        if path.exists():
            path.unlink()
    cache_dir = Path(args.work_root) / f"jit_cache_{kernel_name}_{index:02d}_{role}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    env = {
        **os.environ,
        "PYTHONPATH": f"{repo}:{repo / 'tests'}",
        "DG_JIT_CACHE_DIR": str(cache_dir),
        "DG_USE_LOCAL_VERSION": "0",
    }
    variant = f"{args.run_group_id}_{kernel_name}_{index:02d}_{role}"
    proc = check_cmd(meta["cmd"](args.python, index, variant), cwd=repo, env=env)
    stdout_parts.append(f"RUN {kernel_name} {index:02d} {role}\n{proc.stdout}")
    stderr_parts.append(f"RUN {kernel_name} {index:02d} {role}\n{proc.stderr}")
    return parse_native_row(kernel_name, repo, variant)


def summary_for(role: str, raws: list[dict]) -> dict:
    role_rows = [raw for raw in raws if raw["role"] == role]
    latencies = [raw["latency_ms"] for raw in role_rows]
    tflops_values = [raw["tflops"] for raw in role_rows]
    sol_values = [raw["sol_pct"] for raw in role_rows]
    return {
        "samples": len(role_rows),
        "median_latency_ms": statistics.median(latencies),
        "p10_latency_ms": percentile(latencies, 0.10),
        "p90_latency_ms": percentile(latencies, 0.90),
        "std_latency_ms": statistics.pstdev(latencies),
        "mad_latency_ms": mad(latencies),
        "median_tflops": statistics.median(tflops_values),
        "median_sol_pct": statistics.median(sol_values),
    }


def make_correctness(kernel_name: str, rows: list[tuple[str, dict]], full_gate: dict[str, dict]) -> dict:
    target_failed = sum(0 if row.get("passed") else 1 for _, row in rows)
    full_gate_failed = sum(int(gate["failed_cases"]) for gate in full_gate.values())
    return {
        "accuracy_pass": target_failed == 0 and full_gate_failed == 0,
        "log_path": "accuracy.log",
        "tolerance": (
            "target FP4xFP4 max_diff<0.02; full gate tests/sm120/test_m_grouped_fp8.py "
            "requires contiguous 18/18 and masked 15/15"
        ),
        "failed_samples": target_failed + full_gate_failed,
        "target_kernel": kernel_name,
        "target_failed_samples": target_failed,
        "full_accuracy_gate": {
            role: {
                "pass": gate["pass"],
                "returncode": gate["returncode"],
                "contiguous_passed": gate["contiguous_passed"],
                "contiguous_total": gate["contiguous_total"],
                "masked_passed": gate["masked_passed"],
                "masked_total": gate["masked_total"],
                "failed_cases": gate["failed_cases"],
            }
            for role, gate in full_gate.items()
        },
    }


def write_kernel_record(
    args: argparse.Namespace,
    kernel_name: str,
    hardware: dict,
    rows: list[tuple[str, dict]],
    full_gate: dict[str, dict],
    stdout_parts: list[str],
    stderr_parts: list[str],
) -> None:
    meta = KERNELS[kernel_name]
    run_dir = Path(args.run_root) / kernel_name
    raw_dir = run_dir / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    source = {
        "baseline_commit": args.baseline_commit,
        "candidate_commit": args.candidate_commit,
        "branch": args.branch,
        "kernel_name": meta["kernel_name"],
        "shape": meta["shape"],
        "dtype": "fp4_x_fp4_bf16_out",
        "source_state": (
            f"endpoint A/B over range {args.baseline_commit[:12]}..{args.candidate_commit[:12]}; "
            "full DeepGEMM M-grouped accuracy gate must pass for both roles"
        ),
    }
    harness = {
        "script_path": f"harness/{meta['bench_script']} via harness/kbc_deepgemm_g1_g2_ab.py; source=avo/kbc_deepgemm_g1_g2_ab.py",
        "command": args.command_label,
        "timing_method": (
            f"deep_gemm.testing.bench_kineto: torch profiler schedule warmup=1 active=1, "
            f"num_tests={BENCH_KINETO_NUM_TESTS}, flush_l2=True; native median of "
            f"{meta['native_repeats']} bench_kineto calls per KBC sample"
        ),
        "cuda_graph_enabled": False,
        "graph_launches": 0,
        "pdl_enabled": False,
    }
    cache_policy = {
        "mode": "deep_gemm.testing.bench_kineto flush_l2=True; 8GB memset before each profiled kernel; fresh DG_JIT_CACHE_DIR per KBC sample",
        "workspace_count": 1,
        "workspace_size_bytes": 0,
        "covers_3x_l2": True,
        "rotates_address_each_round": False,
    }
    correctness = make_correctness(kernel_name, rows, full_gate)
    if not correctness["accuracy_pass"]:
        raise RuntimeError(f"{kernel_name} correctness gate failed: {json.dumps(correctness, sort_keys=True)}")

    raw_objects: list[dict] = []
    runs: list[dict] = []
    artifact_lines: list[str] = []
    for index, (role, row) in enumerate(rows):
        run_id = f"{args.run_group_id}-{kernel_name}-{index:02d}-{role}"
        raw = {
            "source": source,
            "hardware": hardware,
            "harness": harness,
            "cache_policy": cache_policy,
            "correctness": correctness,
            "role": role,
            "run_group_id": f"{args.run_group_id}-{kernel_name}",
            "run_id": run_id,
            "timestamp": utc_timestamp(),
            "latency_ms": row["latency_ms"],
            "tflops": row["tflops"],
            "sol_pct": row["sol_pct"],
            "actual_iterations": BENCH_KINETO_NUM_TESTS,
            "native": row["native_row"],
        }
        raw_path = raw_dir / f"{index:02d}_{role}.json"
        raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        raw_sha = sha256_file(raw_path)
        raw_objects.append(raw)
        artifact_lines.append("KBC_RUN_ARTIFACT " + json.dumps(raw, sort_keys=True))
        runs.append(
            {
                "role": role,
                "run_id": run_id,
                "timestamp": raw["timestamp"],
                "latency_ms": raw["latency_ms"],
                "tflops": raw["tflops"],
                "sol_pct": raw["sol_pct"],
                "raw_artifact": f"raw_runs/{raw_path.name}",
                "raw_sha256": raw_sha,
            }
        )

    record = {
        "version": "1.0",
        "name": f"{args.name}_{kernel_name}",
        "source": source,
        "hardware": hardware,
        "harness": harness,
        "cache_policy": cache_policy,
        "measurement": {
            "run_group_id": f"{args.run_group_id}-{kernel_name}",
            "compare_mode": "side_by_side_ab",
            "warmup": 1,
            "iterations": BENCH_KINETO_NUM_TESTS,
            "repeats": args.pairs,
            "native_repeats_per_sample": meta["native_repeats"],
            "max_pair_gap_seconds": 1800,
            "runs": runs,
            "summary": {
                "baseline": summary_for("baseline", raw_objects),
                "candidate": summary_for("candidate", raw_objects),
            },
        },
        "correctness": correctness,
        "artifacts": {
            "stdout": "stdout.log",
            "stderr": "stderr.log",
            "raw_backend_result": "raw_backend_result.json",
            "accuracy": "accuracy.log",
        },
    }
    (run_dir / "benchmark_record.yaml").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (run_dir / "stdout.log").write_text("\n".join(stdout_parts + artifact_lines) + "\n")
    (run_dir / "stderr.log").write_text("\n".join(stderr_parts) + "\n")
    (run_dir / "raw_backend_result.json").write_text(
        json.dumps({"stdouts": ["\n".join(artifact_lines) + "\n"], "successes": [True]}, indent=2, sort_keys=True) + "\n"
    )
    acc_lines = [
        f"accuracy_pass={correctness['accuracy_pass']}",
        f"failed_samples={correctness['failed_samples']}",
        f"tolerance={correctness['tolerance']}",
        "",
        "Full accuracy gate:",
    ]
    for role in ("baseline", "candidate"):
        gate = full_gate[role]
        acc_lines.extend(
            [
                (
                    f"{role}: pass={gate['pass']} returncode={gate['returncode']} "
                    f"contiguous={gate['contiguous_passed']}/{gate['contiguous_total']} "
                    f"masked={gate['masked_passed']}/{gate['masked_total']} "
                    f"failed_cases={gate['failed_cases']}"
                ),
                gate["log"].rstrip(),
                "",
            ]
        )
    acc_lines.append("Target timed samples:")
    for index, (role, row) in enumerate(rows):
        acc_lines.append(
            f"{index:02d} role={role} passed={row.get('passed')} max_diff={row.get('max_diff')} tolerance={row.get('tolerance')}"
        )
    (run_dir / "accuracy.log").write_text("\n".join(acc_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--run-group-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--command-label", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    Path(args.work_root).mkdir(parents=True, exist_ok=True)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    repos = {
        "baseline": prepare_repo(args, "baseline", args.baseline_commit, stdout_parts, stderr_parts),
        "candidate": prepare_repo(args, "candidate", args.candidate_commit, stdout_parts, stderr_parts),
    }
    hardware = collect_gpu_info(args.python, repos["candidate"])

    full_gate = {}
    for role, repo in repos.items():
        gate = run_full_accuracy_gate(args, repo, role)
        full_gate[role] = gate
        log_path = run_root / f"full_accuracy_{role}.log"
        log_path.write_text(gate["log"])
        print(
            json.dumps(
                {
                    "event": "full_accuracy_gate",
                    "role": role,
                    "pass": gate["pass"],
                    "returncode": gate["returncode"],
                    "contiguous": f"{gate['contiguous_passed']}/{gate['contiguous_total']}",
                    "masked": f"{gate['masked_passed']}/{gate['masked_total']}",
                    "failed_cases": gate["failed_cases"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if any(not gate["pass"] for gate in full_gate.values()):
        raise RuntimeError(
            "full DeepGEMM M-grouped accuracy gate failed: "
            + json.dumps(
                {
                    role: {key: value for key, value in gate.items() if key != "log"}
                    for role, gate in full_gate.items()
                },
                sort_keys=True,
            )
        )

    order: list[str] = []
    for _ in range(args.pairs):
        order.extend(["baseline", "candidate"])

    kernel_rows: dict[str, list[tuple[str, dict]]] = {"g1": [], "g2": []}
    for index, role in enumerate(order):
        for kernel_name in ("g1", "g2"):
            row = run_kernel_sample(kernel_name, repos[role], role, index, args, stdout_parts, stderr_parts)
            print(
                json.dumps(
                    {
                        "kernel": kernel_name,
                        "index": index,
                        "role": role,
                        **{k: row[k] for k in ("latency_ms", "tflops", "sol_pct", "passed") if k in row},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            kernel_rows[kernel_name].append((role, row))

    for kernel_name in ("g1", "g2"):
        write_kernel_record(args, kernel_name, hardware, kernel_rows[kernel_name], full_gate, stdout_parts, stderr_parts)

    print(
        json.dumps(
            {
                kernel_name: {
                    role: summary_for(
                        role,
                        [
                            {
                                "role": row_role,
                                "latency_ms": row["latency_ms"],
                                "tflops": row["tflops"],
                                "sol_pct": row["sol_pct"],
                            }
                            for row_role, row in rows
                        ],
                    )
                    for role in ("baseline", "candidate")
                }
                for kernel_name, rows in kernel_rows.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
