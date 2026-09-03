# Port the SM90 fused MegaMoE kernel from NVFP4 to MXFP4

Base: `deepseek-base-optimization` (24eace8)
Branch: `feat/sm90-mxfp4-megamoe` (also pushed to AichenF as `megamoe_mxfp4_dev_m`)

DeepSeek-V4-Flash stores its routed experts in OCP MXFP4, so the H200 fused
MegaMoE kernel needs an MXFP4 weight path. This carries AichenF's NVFP4 kernel
(`megamoe_nvfp4_dev_m`, merged in `e3cf8be`, authorship preserved) plus the
MXFP4 port on top.

## Why it isn't a transliteration

NVFP4's UE4M3 scale has a 3-bit mantissa, so `magnitude * scale` is a real
multiply and the cheapest evaluation is a 128-row table indexed by the scale
byte -- a dependent SMEM load in the mainloop prologue plus 1 KB of SMEM.
MXFP4's E8M0 scale is a pure power of two, so the same product is an exponent
adjustment: multiplying an E4M3 value by 2^k adds k to its 4-bit exponent field,
i.e. adds `(k << 3)` to the byte. The table collapses to eight constant bytes in
immediates and applying a scale group becomes two integer adds in registers.
No table, no SMEM, no dependent load -- and the prefetch pipeline that existed
to hide that load goes away with it.

First attempt used `__vaddus4`/`__vsubus4`/`__vminu4`. nvcc emulates those on
sm_90 and table construction ballooned to ~70 ALU ops, worse than the load it
replaced. The seven nonzero base bytes hold exponent fields 6..9, so for
k in [-5, 6] every byte stays inside [0x08, 0x7c] and a plain 32-bit add is
exact with no carry between lanes. The saturating path is kept for scales
outside that window, which real weights do reach.

## Measured on H20-3e (sm_90), three runs per point

| tiles | MXFP4 vs NVFP4 |
|---|---|
| 8192 | +0.16% |
| 32768 | -5.7% |
| 98304 | -13.3% |
| **196608** (DSv4-Flash L1 shape) | **-19.6%** |
| 393216 | -10.0% |

The win is not uniform. Below ~32k tiles the working set is small enough that
scale traffic is nearly free for both formats and they are a wash. It peaks at
the L1 shape the routed experts actually produce, and falls back at larger
sizes where both saturate DRAM on the packed-weight stream, which is identical
between them.

Scale codes in the benchmark must vary -- an earlier version filled them with a
constant, letting NVFP4's lookup hit the same constant-cache row every time and
hiding the cost the benchmark exists to measure. That version read -10.9% at the
L1 shape instead of -19.6%.

| | |
|---|---|
| dequant precision | **100% exact** vs NVFP4's 74.4% in E4M3's normal range |
| scale folding | 4096/4096 bit-exact, all 256 E8M0 codes x 16 nibbles |
| decode chain | 0/32768 against real `transform_mxfp4_weights_for_mega_moe_sm90` output |
| layout suite | 6/6 |
| compile | sm_90a, 17 wgmma (matches NVFP4); gated out on sm_120a |

## Known limitation

The kernel cannot run on H20. Past the binding it is rejected by
`is_supported_h200_shape()`, whose first constant is `kH200NumSMs = 132`;
H20-3e has 78. The `h200` in the filename is literal. The unmodified NVFP4
kernel refuses H20 identically, so this is inherited, not introduced here --
but it means `tests/test_mxfp4_mega_moe_sm90_correctness.py` needs an H200 to
run. Retargeting is not a constant swap: `kNumSMs` feeds four
`grid_sync`/`nvlink_barrier` instantiations that must match the launched grid
exactly, and a mismatch deadlocks rather than errors.
