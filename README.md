# batched-tiny-linalg (A1 seed repo)

Warp-per-matrix and thread-per-matrix, register-resident batched Cholesky
factor+solve for tiny SPD matrices (n <= 32), hand-written CUDA vs cuSOLVER
batched and Eigen CPU. The claim: at this size there is no launch tax, no
workspace query, no transposed write-back, so the library floor loses.
Literature anchor: Haidar/Abdelfattah/Dongarra (2-4x at n<=32).

## Status

- Full 60-config sweep published (float32 and float64, n=2..32, batch 64 to
  65536); btl wins 58/60 configs against cuSOLVER, both losses published.
- Recorded on an RTX 5060 Laptop (`sm_120`), CUDA 13.2, Eigen 5.0.1; GPU
  times are CUDA-event medians (20 iters, 5 warmup).

## Headline: btl vs cuSOLVER batched

Fused Cholesky factor + solve, SPD batches, ms per batched call, best of the
two btl variants against `potrfBatched` + `potrsBatched`. Selected cells:

| precision | n | batch | best btl | cuSOLVER | btl vs cuSOLVER | winner |
|---|--:|--:|--:|--:|--:|---|
| f32 | 2 | 65536 | 0.014 | 2.32 | 160.10x | btl |
| f32 | 8 | 1024 | 0.015 | 0.082 | 5.49x | btl |
| f32 | 32 | 1024 | 0.098 | 0.179 | 1.82x | btl |
| f64 | 32 | 64 | 0.145 | 0.126 | 0.87x | cusolver |
| f64 | 32 | 1024 | 0.745 | 0.660 | 0.89x | cusolver |
| f64 | 32 | 65536 | 60.9 | 90.2 | 1.48x | btl |

## Key numbers

- btl wins 58/60 configs, by 1.04x to 160x vs cuSOLVER and up to ~826x vs
  single-thread Eigen CPU.
- Kill criterion verdict: cuSOLVER batched wins n=32 float64 at batch 64 and
  1024 (0.126 ms vs 0.145 ms; 0.660 ms vs 0.745 ms). Condition: fp64 +
  largest n + small/medium batch, where the launch-latency advantage is
  amortized away.
- Floors: cuSOLVER ~70 µs per potrf+potrs pair (launch + pointer-array
  indirection); btl ~14 µs (one kernel launch). That floor is the entire
  game at small batch and small n.

## Honest limits

- The two cuSOLVER wins are published in the table, per the kill criterion.
- Eigen CPU wins the trivial cells (n <= 3, batch = 64) where the GPU
  cannot get below launch latency.
- The warp-per-matrix variant loses to thread-per-matrix for float64 at
  batch 65536 (n=12: 7.22 ms vs 1.07 ms): shuffle-heavy factor path plus
  register pressure on 64-bit lanes. Both variants kept; optimization
  deferred.

## Build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
./build/bench_cholesky            # writes bench_cholesky_results.json
```

CMake auto-selects `g++-15` as the CUDA host compiler when the default g++
is >= 16 (nvcc 13.2 rejects it); no flags needed.

## Repository layout

- `include/btl/`: header-only kernels (device code)
- `src/`: host wrappers, non-header pieces
- `tests/`: correctness (residual-based, vs Eigen)
- `benchmarks/`: timing harness, results persisted to JSON

Standalone seed repo: no dependencies beyond the CUDA toolkit and Eigen.
Other projects vendor FROM here (`include/btl/cholesky.cuh` is designed to
be copied as a single file); it depends on nothing itself.

## Deeper reading

- `docs/RUNGS.md`: the full record: claim, kill criterion, protocol, both
  30-row result tables verbatim, the verdict, and the observations.
