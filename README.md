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
```

## Standalone
Seed repo — no dependencies beyond CUDA toolkit + Eigen. This is the repo
other projects vendor FROM (`include/btl/cholesky.cuh` is designed to be
copied as a single file); it depends on nothing itself.

## TODO (fill in one at a time, in this order)
1. [ ] `include/btl/cholesky.cuh` — device: factor, solve, factor_solve
2. [ ] `src/alloc.cuh` — MemMode {device, pinned, unified}, plan alloc/free/upload/download
3. [ ] `tests/test_cholesky.cpp` — residual check across n and batch
4. [ ] `benchmarks/bench_cholesky.cpp` — vs cuSOLVER batched + Eigen; JSON out
5. [ ] README results table (measured, with kill-criterion condition)
