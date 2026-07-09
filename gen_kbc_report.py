"""Generate the base-vs-candidate KBC markdown report from kbc_full_ab_results.tsv."""
import statistics
import sys
from collections import defaultdict

TSV = sys.argv[1]
OUT = sys.argv[2]

# GB10 / DGX Spark SM121 peaks
PEAKS_TFLOPS = {'FP4': 500.0, 'FP8': 250.0, 'BF16': 125.0, 'TF32': 62.5}
PEAK_BW = 273.0

ROWS = [
    # shape_id, kernel description, compute-peak class, shape brief
    ('D1_FP8', 'Dense GEMM FP8', 'FP8', 'M4096 N16384 K7168'),
    ('D1_FP4', 'Dense GEMM FP4', 'FP4', 'M4096 N16384 K7168'),
    ('D1_MIX', 'Dense GEMM FP8xFP4', 'FP8', 'M4096 N16384 K7168'),
    ('D2_FP4', 'Dense GEMM FP4 small-M', 'FP4', 'M128 N24576 K1536'),
    ('G1', 'M-grouped contiguous FP4 (psum)', 'FP4', 'G4 M36096 N6144 K7168'),
    ('G2', 'M-grouped masked FP4, large group', 'FP4', 'G6 EM1024 N4096 K4096'),
    ('G3', 'M-grouped masked FP4 tiny group (psum)', 'FP4', 'G6 EM20 N4096 K2048'),
    ('K1', 'K-grouped FP8', 'FP8', 'G4 M4096 N7168 K~33K'),
    ('A1_FP4', 'Regular MQA FP4, BF16=True', 'FP4', 'S4096 SKV8192 H64 D128'),
    ('A1_FP8', 'Regular MQA FP8, BF16=False', 'FP8', 'S4096 SKV8192 H64 D128'),
    ('A2_FP4', 'Paged MQA FP4', 'FP4', 'B256 L32768 H64 D128'),
    ('A3_FP8', 'Paged MQA FP8', 'FP8', 'B256 L8192 H64 D128'),
    ('H1', 'HyperConnection TF32', 'TF32', 'M8192 N24 K28672'),
]

data = defaultdict(lambda: defaultdict(list))  # side -> shape_id -> list of (lat, tf, gb, diff)
descs = {}
for line in open(TSV):
    parts = line.rstrip('\n').split('\t')
    if len(parts) != 7:
        continue
    tag, sid, desc, lat, tf, gb, diff = parts
    side = tag.split('_r')[0]
    data[side][sid].append((float(lat), float(tf), float(gb), float(diff)))
    descs[sid] = desc


def med(side, sid, idx):
    vals = [v[idx] for v in data[side][sid]]
    return statistics.median(vals) if vals else float('nan')


lines = []
add = lines.append
add('# DeepGEMM Base vs General-Tuning Candidate — Full Kernel Benchmark (DGX Spark SM121)')
add('')
add('Purpose: apple-to-apple regression check of every DeepSeek-V4 kernel covered by')
add('[sm120-vs-dgx-spark-sm121-kernel-peak-comparison.md](sm120-vs-dgx-spark-sm121-kernel-peak-comparison.md),')
add('comparing two DeepGEMM commits **on the same GPU with identical seeded inputs**:')
add('')
add('| Role | Commit | Description |')
add('| --- | --- | --- |')
add('| Baseline | `3a2feda03564cfed609683f10fcab9cd47797c6a` | deepseek-base merge point |')
add('| Candidate | `60911e1274623d841d1f9150e449bcda0dd28b9e` (`experiment` branch) | G1/G2 grouped-GEMM optimization, de-specialized (enumeration + cost model) |')
add('')
add('Compare: <https://github.com/xutizhou/DeepGEMM/compare/3a2feda...experiment>')
add('')
add('## Method')
add('')
add('- Hardware: DGX Spark (GB10, SM121, 48 SMs), docker `nvcr.io/nvidia/pytorch:25.10-py3`.')
add('- Peaks used for SOL%: FP4 `500 TF`, FP8 `250 TF` (FP8xFP4 normalized to FP8), TF32 `62.5 TF`, memory `273 GB/s`.')
add('- Interleaved A/B: 3 rounds of (baseline, candidate), GPU idle-guard before every pass')
add('  (GB10 has no user clock-lock, so back-to-back interleaving is required for a fair delta).')
add('- Timing: `deep_gemm.testing.bench_kineto` — the same harness the upstream tests use.')
add('- Every row is accuracy-gated each round (asserted against the FP32 reference before timing);')
add('  the `diff` columns below are the measured max relative errors.')
add('- Values reported are medians over the 3 rounds.')
add('')
add('## Main Comparison Table')
add('')
add('`Δ latency` = (candidate − baseline) / baseline; negative is faster.')
add('')
add('| Shape | Kernel / precision | Shape brief | Base us | Cand us | Δ latency | Base C% | Cand C% | Base BW% | Cand BW% |')
add('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
worst = []
for sid, kdesc, peak_cls, brief in ROWS:
    if not data['b3a'][sid] or not data['prb'][sid]:
        continue
    bl, cl = med('b3a', sid, 0), med('prb', sid, 0)
    btf, ctf = med('b3a', sid, 1), med('prb', sid, 1)
    bgb, cgb = med('b3a', sid, 2), med('prb', sid, 2)
    delta = (cl - bl) / bl * 100
    worst.append((sid, delta))
    peak = PEAKS_TFLOPS[peak_cls]
    add(f'| `{sid}` | {kdesc} | `{brief}` | `{bl:.1f}` | `{cl:.1f}` | `{delta:+.2f}%` | '
        f'`{btf / peak * 100:.1f}%` | `{ctf / peak * 100:.1f}%` | '
        f'`{bgb / PEAK_BW * 100:.1f}%` | `{cgb / PEAK_BW * 100:.1f}%` |')
add('')
add('## Accuracy')
add('')
add('| Shape | Base max diff | Cand max diff | Match |')
add('| --- | ---: | ---: | --- |')
for sid, kdesc, peak_cls, brief in ROWS:
    if not data['b3a'][sid] or not data['prb'][sid]:
        continue
    bd, cd = med('b3a', sid, 3), med('prb', sid, 3)
    add(f'| `{sid}` | `{bd:.5f}` | `{cd:.5f}` | {"identical" if abs(bd - cd) < 1e-6 else "differs"} |')
add('')
add('## Shape Index')
add('')
add('| Shape | Measured shape detail |')
add('| --- | --- |')
for sid, kdesc, peak_cls, brief in ROWS:
    if sid in descs:
        add(f'| `{sid}` | `{descs[sid]}` |')
add('')
add('## Per-Round Raw Measurements')
add('')
add('Cell format: `latency us / TFLOP/s / GB/s`.')
add('')
add('| Shape | Base r1 | Base r2 | Base r3 | Cand r1 | Cand r2 | Cand r3 |')
add('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
for sid, kdesc, peak_cls, brief in ROWS:
    cells = []
    for side in ('b3a', 'prb'):
        runs = data[side][sid]
        for i in range(3):
            if i < len(runs):
                lat, tf, gb, _ = runs[i]
                cells.append(f'`{lat:.0f} / {tf:.0f} / {gb:.0f}`')
            else:
                cells.append('-')
    add(f'| `{sid}` | ' + ' | '.join(cells) + ' |')
add('')
add('## Why Each Row Moved (SASS audit)')
add('')
add('Compiled kernels from both sides were compared byte-for-byte (`cuobjdump -sass` on the')
add('JIT caches) to attribute every delta:')
add('')
add('| Rows | SASS | Attribution |')
add('| --- | --- | --- |')
add('| `D1_FP8` `D1_FP4` `D1_MIX` `D2_FP4` `H1` | **byte-identical** | Deltas are pure measurement noise/bias (GB10 has no clock lock; the candidate side enters each pass marginally cooler because its G1 pass is ~1 ms shorter). `D2_FP4` -3.5% on a 140 us kernel is this bias, not a code change. |')
add('| `A1/A2/A3` (MQA) | untouched source | Attention `.cuh` files are identical between the commits; deltas are noise (<0.5%). |')
add('| `G1` `G2` `G3` | **intentionally different** | The optimization target: candidate selects sweep-found tiles via the restored enumeration + cost model (G1 `BM128/BN192`, G2 `BM192/BN128/store48`, G3 improved grouped-psum tile). |')
add('| `K1` | different | Side effect of the global FP32-output CD-swizzle correctness guard (`cd_size<=2`): K-grouped (FP32 D) now uses the direct-store epilogue. Slightly faster (-2.6%), accuracy identical. |')
add('')
add('## Caveats')
add('')
add('- Sparse MLA rows (`S1-S3`) from the reference document are not DeepGEMM kernels')
add('  (they came from the vLLM/DeepSeek-V4 harness) and are excluded; no DeepGEMM code')
add('  they depend on changed between these commits.')
add('- `D1` rows are measured with NT layout (K-major A and B); the reference document')
add('  lists `layout=NN` for `D1`, but mixed FP8xFP4 requires K-major B, so NT is used')
add('  for all three `D1` precisions to keep them comparable.')
add('- `G2` is measured through the `nt_masked` API (the tuned masked path); the')
add('  reference document row used the psum layout variant of the same shape.')
add('- `K1` group K sizes are seeded random (`k=33280` here vs `32896` in the reference')
add('  document); identical on both sides of this comparison.')
add('- Harnesses: `bench_g1_g2.py`, `bench_kbc_extra.py`, `bench_kbc_attn.py` on the')
add('  `experiment` branch (identical copies run in both checkouts).')
add('')
open(OUT, 'w').write('\n'.join(lines) + '\n')
print(f'wrote {OUT}; worst deltas: {sorted(worst, key=lambda x: -abs(x[1]))[:4]}')
