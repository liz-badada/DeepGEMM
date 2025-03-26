## deepgemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked

### tests/test_core.py:test_m_grouped_gemm_masked
```mermaid
sequenceDiagram
    participant Test as test_core.py
    participant GEMM as m_grouped_gemm.py
    participant JIT as jit_tuner
    participant CUDA as CUDA Kernel

    Note over Test: test_m_grouped_gemm_masked()
    
    Test->>Test: construct_grouped()<br/>Create input tensors
    Note over Test: Generate:<br/>- x_fp8, lhs_scales<br/>- y_fp8, rhs_scales<br/>- out<br/>- masked_m
    
    Test->>GEMM: m_grouped_gemm_fp8_fp8_bf16_nt_masked()
    
    GEMM->>GEMM: Parameter validation<br/>- Type checking<br/>- Shape checking<br/>- Contiguity checking
    
    GEMM->>GEMM: Prepare lhs scales<br/>get_col_major_tma_aligned_tensor()
    
    GEMM->>GEMM: Get optimal configuration<br/>get_best_configs()<br/>- num_sms<br/>- block_m<br/>- block_n<br/>- num_stages<br/>- tma_multicast_config<br/>- smem_size
    
    GEMM->>JIT: compile_and_tune()<br/>Compile and tune CUDA kernel
    
    JIT->>CUDA: Compile CUDA code
    
    JIT->>CUDA: Execute kernel<br/>Parameters:<br/>- lhs, lhs_scales<br/>- rhs, rhs_scales<br/>- out<br/>- masked_m<br/>- m<br/>- stream<br/>- num_sms<br/>- smem_size
    
    CUDA-->>Test: Return results
```

### input/output (example: num_groups=4, m=256, n=2048, k=7168)
| Parameter | Type | Shape | Dtype | Description | Example |
|-----------|------|--------|--------|-------------|------------------------------------------------|
| lhs | Tuple(torch.Tensor, torch.Tensor) | ([num_groups, m_max, k], [num_groups, m_max, ⌈k/128⌉]) | (torch.float8_e4m3fn, torch.float32) | input & scale | ([4, 256, 7168], [4, 256, 56]) |
| rhs | Tuple(torch.Tensor, torch.Tensor) | ([num_groups, n, k], [num_groups, ⌈n/128⌉, ⌈k/128⌉]) | (torch.float8_e4m3fn, torch.float32) | weight & scale | ([4, 2048, 7168], [4, 32, 56]) |
| out | torch.Tensor | [num_groups, m_max, n] | torch.bfloat16 | output | [4, 256, 2048] |
| masked_m | torch.Tensor | [num_groups] | torch.int32 | actual rows to compute for each group | [4] |
| expected_m | int | - | - | hint for M dimension expectation | min(mean(masked_m) + 1, m) |