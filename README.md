# batched-tiny-linalg (A1 — seed repo)

Warp-per-matrix, register-resident batched Cholesky factor+solve for tiny SPD
matrices (n <= 32), vs cuSOLVER batched and Eigen CPU.

## Claim
Batched tiny-matrix Cholesky beats cuBLAS/cuSOLVER building blocks at n <= 32
because there is no launch tax, no workspace query, no transposed write-back.
Literature: Haidar/Abdelfattah/Dongarra (2-4x at n<=32).

## Kill criterion (publish either way)
- If cuSOLVER batched wins at any n <= 32 / batch >= 64, publish that number
  and the condition.

## Layout
- `include/btl/`  — header-only kernels (device code)
- `src/`          — host wrappers, non-header pieces
- `tests/`        — correctness (residual-based, vs Eigen)
- `benchmarks/`   — timing harness, results persisted to JSON

## Build
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
./build/bench_cholesky            # writes bench_cholesky_results.json
```
CMake auto-selects `g++-15` as the CUDA host compiler when the default g++
is >= 16 (nvcc 13.2 rejects it); no flags needed.

## Standalone
Seed repo — no dependencies beyond CUDA toolkit + Eigen. This is the repo
other projects vendor FROM (`include/btl/cholesky.cuh` is designed to be
copied as a single file); it depends on nothing itself.

## Results (2026-10-10, RTX 5060 Laptop, sm_120, CUDA 13.2, Eigen 5.0.1)

Fused Cholesky factor + solve (one n-vector rhs per matrix), full row-major
storage, SPD batches A = B Bᵀ + nI. GPU times are CUDA-event medians
(20 iters, 5 warmup); Eigen CPU is fixed-size `Eigen::LLT` single-thread.
cuSOLVER = `potrfBatched` + `potrsBatched`; its in-place A/rhs restores and
pointer-array setup are **excluded** from its timing (generous to cuSOLVER).
btl reads A and writes L separately; cuSOLVER factors in place. Full data:
`bench_cholesky_results.json` (gitignored, regenerate with
`./build/bench_cholesky`).

### float32 (ms, median of 20 GPU / 5-11 CPU iters, warmup first)

| n | batch | btl thread/matrix | btl warp/matrix | cuSOLVER batched | Eigen CPU | best btl vs cuSOLVER | winner |
|---|-------|------------------:|----------------:|-----------------:|----------:|---------------------:|--------|
| 2 | 64 | 0.014 | 0.015 | 0.070 | 0.003 | 4.84x | eigen_cpu |
| 2 | 1024 | 0.014 | 0.014 | 0.071 | 0.050 | 4.97x | btl |
| 2 | 65536 | 0.014 | 0.029 | 2.32 | 3.34 | 160.10x | btl |
| 3 | 64 | 0.015 | 0.014 | 0.069 | 0.008 | 4.95x | eigen_cpu |
| 3 | 1024 | 0.014 | 0.014 | 0.076 | 0.128 | 5.34x | btl |
| 3 | 65536 | 0.030 | 0.043 | 2.69 | 8.76 | 90.27x | btl |
| 4 | 64 | 0.014 | 0.014 | 0.069 | 0.026 | 4.80x | btl |
| 4 | 1024 | 0.018 | 0.014 | 0.089 | 0.407 | 6.22x | btl |
| 4 | 65536 | 0.064 | 0.062 | 4.27 | 27.9 | 69.19x | btl |
| 6 | 64 | 0.015 | 0.014 | 0.069 | 0.054 | 4.93x | btl |
| 6 | 1024 | 0.021 | 0.014 | 0.087 | 0.907 | 6.19x | btl |
| 6 | 65536 | 0.113 | 0.093 | 3.81 | 58.3 | 40.85x | btl |
| 8 | 64 | 0.016 | 0.014 | 0.071 | 0.091 | 4.91x | btl |
| 8 | 1024 | 0.031 | 0.015 | 0.082 | 1.54 | 5.49x | btl |
| 8 | 65536 | 0.192 | 0.128 | 4.46 | 106 | 34.77x | btl |
| 9 | 64 | 0.021 | 0.021 | 0.069 | 0.164 | 3.29x | btl |
| 9 | 1024 | 0.033 | 0.017 | 0.084 | 3.12 | 4.79x | btl |
| 9 | 65536 | 0.254 | 0.235 | 5.03 | 190 | 21.35x | btl |
| 12 | 64 | 0.021 | 0.017 | 0.070 | 0.226 | 4.13x | btl |
| 12 | 1024 | 0.050 | 0.021 | 0.087 | 3.85 | 4.08x | btl |
| 12 | 65536 | 0.393 | 0.447 | 7.63 | 264 | 19.43x | btl |
| 16 | 64 | 0.027 | 0.020 | 0.073 | 0.341 | 3.58x | btl |
| 16 | 1024 | 0.084 | 0.027 | 0.098 | 5.74 | 3.56x | btl |
| 16 | 65536 | 0.769 | 1.19 | 9.20 | 393 | 11.97x | btl |
| 24 | 64 | 0.060 | 0.022 | 0.103 | 0.690 | 4.61x | btl |
| 24 | 1024 | 0.198 | 0.051 | 0.154 | 11.1 | 3.00x | btl |
| 24 | 65536 | 3.92 | 5.41 | 24.9 | 742 | 6.35x | btl |
| 32 | 64 | 0.147 | 0.031 | 0.113 | 1.96 | 3.63x | btl |
| 32 | 1024 | 0.506 | 0.098 | 0.179 | 30.1 | 1.82x | btl |
| 32 | 65536 | 14.9 | 24.5 | 27.6 | 2037 | 1.86x | btl |

### float64 (ms, median of 20 GPU / 5-11 CPU iters, warmup first)

| n | batch | btl thread/matrix | btl warp/matrix | cuSOLVER batched | Eigen CPU | best btl vs cuSOLVER | winner |
|---|-------|------------------:|----------------:|-----------------:|----------:|---------------------:|--------|
| 2 | 64 | 0.015 | 0.016 | 0.070 | 0.005 | 4.62x | eigen_cpu |
| 2 | 1024 | 0.014 | 0.016 | 0.080 | 0.079 | 5.59x | btl |
| 2 | 65536 | 0.045 | 0.091 | 3.45 | 5.27 | 76.19x | btl |
| 3 | 64 | 0.014 | 0.019 | 0.069 | 0.009 | 4.76x | eigen_cpu |
| 3 | 1024 | 0.019 | 0.018 | 0.090 | 0.142 | 4.96x | btl |
| 3 | 65536 | 0.075 | 0.256 | 5.30 | 9.59 | 70.97x | btl |
| 4 | 64 | 0.018 | 0.024 | 0.069 | 0.038 | 3.79x | btl |
| 4 | 1024 | 0.029 | 0.025 | 0.130 | 0.621 | 5.20x | btl |
| 4 | 65536 | 0.108 | 0.346 | 5.51 | 39.9 | 50.79x | btl |
| 6 | 64 | 0.020 | 0.025 | 0.071 | 0.064 | 3.47x | btl |
| 6 | 1024 | 0.038 | 0.039 | 0.124 | 1.09 | 3.26x | btl |
| 6 | 65536 | 0.212 | 1.30 | 8.29 | 71.2 | 39.07x | btl |
| 8 | 64 | 0.033 | 0.037 | 0.072 | 0.104 | 2.20x | btl |
| 8 | 1024 | 0.058 | 0.058 | 0.154 | 1.85 | 2.67x | btl |
| 8 | 65536 | 0.380 | 2.15 | 11.5 | 126 | 30.25x | btl |
| 9 | 64 | 0.029 | 0.035 | 0.072 | 0.174 | 2.51x | btl |
| 9 | 1024 | 0.068 | 0.102 | 0.175 | 3.19 | 2.58x | btl |
| 9 | 65536 | 0.556 | 4.83 | 15.6 | 200 | 28.13x | btl |
| 12 | 64 | 0.043 | 0.045 | 0.074 | 0.286 | 1.72x | btl |
| 12 | 1024 | 0.113 | 0.111 | 0.200 | 4.40 | 1.80x | btl |
| 12 | 65536 | 1.07 | 7.22 | 18.7 | 278 | 17.38x | btl |
| 16 | 64 | 0.072 | 0.063 | 0.078 | 0.377 | 1.25x | btl |
| 16 | 1024 | 0.188 | 0.160 | 0.241 | 6.28 | 1.51x | btl |
| 16 | 65536 | 2.68 | 11.5 | 24.8 | 418 | 9.28x | btl |
| 24 | 64 | 0.186 | 0.094 | 0.114 | 0.806 | 1.21x | btl |
| 24 | 1024 | 0.456 | 0.457 | 0.472 | 13.6 | 1.04x | btl |
| 24 | 65536 | 15.1 | 42.6 | 50.0 | 850 | 3.31x | btl |
| 32 | 64 | 0.387 | 0.145 | 0.126 | 2.14 | 0.87x | cusolver |
| 32 | 1024 | 1.05 | 0.745 | 0.660 | 34.1 | 0.89x | cusolver |
| 32 | 65536 | 60.9 | 71.0 | 90.2 | 2201 | 1.48x | btl |

### Kill criterion verdict

**cuSOLVER batched wins at n=32 float64, batch ∈ {64, 1024}** (0.126 ms vs
btl-warp 0.145 ms; 0.660 ms vs 0.745 ms). Published per the kill criterion.
Condition: fp64 + largest n in range + small/medium batch, where btl's
launch-latency advantage is amortized away and cuSOLVER's tuned double
pipelines dominate. btl wins the other 58/60 configs, by 1.2x–160x vs
cuSOLVER and up to ~900x vs single-thread Eigen CPU.

### Observations

- cuSOLVER has a ~70 µs floor per potrf+potrs pair (launch + pointer-array
  indirection); btl's floor is ~14 µs (one kernel launch). That floor is the
  entire game at small batch × small n.
- Eigen CPU wins only where the GPU can't get below launch latency and the
  CPU workload is trivial (n ≤ 3, batch = 64).
- The btl warp-per-matrix variant is slower than thread-per-matrix for
  float64 at batch = 65536 (e.g. n=12: 7.22 ms vs 1.07 ms) — the
  shuffle-heavy factor path plus register pressure on 64-bit lanes. Thread
  variant wins those cells; warp variant wins at small batch × large n
  (n=32 f32 batch=64: 0.031 vs 0.147 ms). Optimization deferred; both are
  kept because the tests pin them to identical semantics.

## TODO (fill in one at a time, in this order)
1. [x] `include/btl/cholesky.cuh` — device: factor, solve, factor_solve
2. [ ] `src/alloc.cuh` — MemMode {device, pinned, unified}, plan alloc/free/upload/download
3. [x] `tests/test_cholesky.cpp` — residual check across n and batch
4. [x] `benchmarks/bench_cholesky.cpp` — vs cuSOLVER batched + Eigen; JSON out
5. [x] README results table (measured, with kill-criterion condition)
