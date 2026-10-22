#pragma once
// btl/cholesky.cuh — batched Cholesky factor + solve for tiny SPD matrices.
//
// Header-only, depends on nothing but the CUDA runtime headers. This file is
// designed to be vendored as a single file into other projects.
//
// Storage: full row-major n x n per matrix, batch-index-major:
//   matrix b, row r, col c  ->  A[b * N * N + r * N + c]
// L is written as the full n x n row-major matrix with the strict upper
// triangle zeroed. rhs / x are one n-vector per matrix: rhs[b * N + i].
//
// Positive-definiteness reporting (LAPACK-style): info[b] == 0 on success;
// info[b] == k (1-based) means pivot k-1 was not positive, i.e. matrix b is
// not numerically SPD. On failure the kernel keeps running (with a safe
// fallback pivot) so results stay finite, but L/x for that matrix are
// garbage and must not be consumed. info is never silently skipped: every
// factor kernel writes info[b] for every b in [0, batch).
//
// Two execution strategies, same semantics:
//   *_thread_per_matrix — one thread factors/solves one whole matrix.
//     Correctness anchor; register-resident only for small N (the N*N tile
//     spills to local memory for large N).
//   *_warp_per_matrix   — one pow2 sub-warp tile per matrix (tile = next
//     pow2 >= N, capped at 32), lane r owns row r. The matrix lives in
//     registers spread across the tile; cross-lane traffic is __shfl_sync.
//
// N is a compile-time template parameter (full unroll); T is float or double.

#include <cuda_runtime.h>

namespace btl {

namespace detail {

template <int N>
struct tile_traits {
    // Sub-warp tile width: smallest power of two >= N, capped at 32.
    static constexpr int TILE = N <= 2 ? 2 : N <= 4 ? 4 : N <= 8 ? 8
                                        : N <= 16 ? 16 : 32;
};

// Common per-thread setup for tile-per-matrix kernels. Returns the lane's
// tile mask; all lanes of the tile run the full algorithm (clamped threads
// with matrix id >= batch operate on a duplicate of the last matrix and are
// masked off at store time) so every __shfl_sync / __ballot_sync stays
// convergent.
template <int TILE>
__device__ __forceinline__ unsigned tile_setup(int batch, int& m, int& lane,
                                               bool& store) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    m = tid / TILE;
    lane = tid % TILE;
    store = (m < batch);
    if (!store) m = batch - 1;  // batch >= 1 guaranteed by host-side guard
    const unsigned laneid = threadIdx.x & 31u;
    if constexpr (TILE == 32) {
        return 0xffffffffu;
    } else {
        const unsigned tile_base = (laneid / TILE) * TILE;
        return ((1u << TILE) - 1u) << tile_base;
    }
}

// In-register Cholesky of the matrix distributed one row per lane.
// On entry row[j] holds A[min(lane,N-1)][j]. On exit row[j] holds
// L[lane][j] for j <= lane (garbage above the diagonal on valid lanes).
// Returns 0 on success, else k+1 where k is the first non-positive pivot
// (uniform across the tile). sqrt/shuffle path is fully unrolled.
template <typename T, int N, int TILE>
__device__ __forceinline__ int factor_tile(unsigned tmask, int lane, T* row) {
    int fail = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        if (lane == k && !(row[k] > T(0))) {
            fail = k + 1;
            row[k] = T(1);  // fallback pivot: keep results finite
        }
        const T lkk = sqrt(__shfl_sync(tmask, row[k], k, TILE));
        const T lrk = row[k] / lkk;  // L[lane][k] on lanes >= k
        if (lane >= k && lane < N) row[k] = lrk;
#pragma unroll
        for (int j = k + 1; j < N; ++j) {
            const T ljk = __shfl_sync(tmask, lrk, j, TILE);  // L[j][k]
            if (lane >= j && lane < N) row[j] -= lrk * ljk;
        }
    }
    return fail;
}

// In-register triangular solves L y = b, L^T x = y. b is this lane's rhs
// component; row[] holds L as produced by factor_tile. Returns this lane's
// x component. Every lane also records the full y/x vectors (broadcast via
// shuffles) since each step's dot products need them locally.
template <typename T, int N, int TILE>
__device__ __forceinline__ T solve_tile(unsigned tmask, int lane,
                                        const T* row, T b) {
    T ys[N];
    T y = T(0);
#pragma unroll
    for (int k = 0; k < N; ++k) {
        T yk = T(0);
        if (lane == k) {
            T acc = b;
#pragma unroll
            for (int j = 0; j < k; ++j) acc -= row[j] * ys[j];
            yk = acc / row[k];
        }
        yk = __shfl_sync(tmask, yk, k, TILE);
        ys[k] = yk;
        if (lane == k) y = yk;
    }
    // Backward substitution: L[j][k] lives on lane j (its row[k]), so lane k
    // gathers the dot product through shuffles. ~N^2/2 shuffles, all
    // convergent across the tile.
    T xs[N];
    T xv = T(0);
#pragma unroll
    for (int k = N - 1; k >= 0; --k) {
        T acc = y;
#pragma unroll
        for (int j = k + 1; j < N; ++j) {
            const T ljk = __shfl_sync(tmask, row[k], j, TILE);
            if (lane == k) acc -= ljk * xs[j];
        }
        T xk = (lane == k) ? acc / row[k] : T(0);
        xk = __shfl_sync(tmask, xk, k, TILE);
        xs[k] = xk;
        if (lane == k) xv = xk;
    }
    return xv;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Thread-per-matrix kernels (correctness anchor)
// ---------------------------------------------------------------------------

template <typename T, int N>
__global__ void factor_thread_per_matrix(const T* __restrict__ A,
                                         T* __restrict__ L, int batch,
                                         int* __restrict__ info) {
    static_assert(N >= 2 && N <= 32);
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const T* a = A + (size_t)b * N * N;
    T* l = L + (size_t)b * N * N;
    T m[N * N];
#pragma unroll
    for (int i = 0; i < N * N; ++i) m[i] = a[i];
    int fail = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        T piv = m[k * N + k];
        if (!(piv > T(0))) {
            if (!fail) fail = k + 1;
            piv = T(1);  // fallback pivot: keep results finite
        }
        const T lkk = sqrt(piv);
        m[k * N + k] = lkk;
#pragma unroll
        for (int r = k + 1; r < N; ++r) {
            const T lrk = m[r * N + k] / lkk;
            m[r * N + k] = lrk;
#pragma unroll
            for (int j = k + 1; j <= r; ++j) m[r * N + j] -= lrk * m[j * N + k];
        }
    }
#pragma unroll
    for (int r = 0; r < N; ++r)
#pragma unroll
        for (int j = 0; j < N; ++j) l[r * N + j] = (j <= r) ? m[r * N + j] : T(0);
    info[b] = fail;
}

// Solves L L^T x = rhs given L from factor_*_per_matrix. No info: callers
// must not solve on a factorization that reported failure.
template <typename T, int N>
__global__ void solve_thread_per_matrix(const T* __restrict__ L,
                                        const T* __restrict__ rhs,
                                        T* __restrict__ x, int batch) {
    static_assert(N >= 2 && N <= 32);
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const T* l = L + (size_t)b * N * N;
    const T* r = rhs + (size_t)b * N;
    T* xo = x + (size_t)b * N;
    T y[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
        T acc = r[k];
#pragma unroll
        for (int j = 0; j < k; ++j) acc -= l[k * N + j] * y[j];
        y[k] = acc / l[k * N + k];
    }
#pragma unroll
    for (int k = N - 1; k >= 0; --k) {
        T acc = y[k];
#pragma unroll
        for (int j = k + 1; j < N; ++j) acc -= l[j * N + k] * y[j];
        y[k] = acc / l[k * N + k];
    }
#pragma unroll
    for (int k = 0; k < N; ++k) xo[k] = y[k];
}

template <typename T, int N>
__global__ void factor_solve_thread_per_matrix(const T* __restrict__ A,
                                               const T* __restrict__ rhs,
                                               T* __restrict__ L,
                                               T* __restrict__ x, int batch,
                                               int* __restrict__ info) {
    static_assert(N >= 2 && N <= 32);
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const T* a = A + (size_t)b * N * N;
    T* l = L + (size_t)b * N * N;
    const T* r = rhs + (size_t)b * N;
    T* xo = x + (size_t)b * N;
    T m[N * N];
#pragma unroll
    for (int i = 0; i < N * N; ++i) m[i] = a[i];
    int fail = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        T piv = m[k * N + k];
        if (!(piv > T(0))) {
            if (!fail) fail = k + 1;
            piv = T(1);
        }
        const T lkk = sqrt(piv);
        m[k * N + k] = lkk;
#pragma unroll
        for (int rr = k + 1; rr < N; ++rr) {
            const T lrk = m[rr * N + k] / lkk;
            m[rr * N + k] = lrk;
#pragma unroll
            for (int j = k + 1; j <= rr; ++j) m[rr * N + j] -= lrk * m[j * N + k];
        }
    }
#pragma unroll
    for (int rr = 0; rr < N; ++rr)
#pragma unroll
        for (int j = 0; j < N; ++j)
            l[rr * N + j] = (j <= rr) ? m[rr * N + j] : T(0);
    info[b] = fail;
    T y[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
        T acc = r[k];
#pragma unroll
        for (int j = 0; j < k; ++j) acc -= m[k * N + j] * y[j];
        y[k] = acc / m[k * N + k];
    }
#pragma unroll
    for (int k = N - 1; k >= 0; --k) {
        T acc = y[k];
#pragma unroll
        for (int j = k + 1; j < N; ++j) acc -= m[j * N + k] * y[j];
        y[k] = acc / m[k * N + k];
    }
#pragma unroll
    for (int k = 0; k < N; ++k) xo[k] = y[k];
}

// ---------------------------------------------------------------------------
// Warp-per-matrix kernels (the claim): one sub-warp tile per matrix,
// register-resident, lane r owns row r.
// ---------------------------------------------------------------------------

template <typename T, int N>
__global__ void factor_warp_per_matrix(const T* __restrict__ A,
                                       T* __restrict__ L, int batch,
                                       int* __restrict__ info) {
    static_assert(N >= 2 && N <= 32);
    if (batch <= 0) return;
    constexpr int TILE = detail::tile_traits<N>::TILE;
    int m, lane;
    bool store;
    const unsigned tmask = detail::tile_setup<TILE>(batch, m, lane, store);
    const int r = min(lane, N - 1);
    const T* a = A + (size_t)m * N * N + r * N;
    T row[N];
#pragma unroll
    for (int j = 0; j < N; ++j) row[j] = a[j];

    const int fail = detail::factor_tile<T, N, TILE>(tmask, lane, row);

    const unsigned bad = __ballot_sync(tmask, fail != 0 && store);
    const int fv = __shfl_sync(tmask, fail, bad ? (__ffs(bad) - 1) : 0, TILE);
    if (store && lane == 0) info[m] = bad ? fv : 0;
    if (store && lane < N) {
        T* l = L + (size_t)m * N * N + (size_t)lane * N;
#pragma unroll
        for (int j = 0; j < N; ++j) l[j] = (j <= lane) ? row[j] : T(0);
    }
}

template <typename T, int N>
__global__ void solve_warp_per_matrix(const T* __restrict__ L,
                                      const T* __restrict__ rhs,
                                      T* __restrict__ x, int batch) {
    static_assert(N >= 2 && N <= 32);
    if (batch <= 0) return;
    constexpr int TILE = detail::tile_traits<N>::TILE;
    int m, lane;
    bool store;
    const unsigned tmask = detail::tile_setup<TILE>(batch, m, lane, store);
    const int r = min(lane, N - 1);
    const T* l = L + (size_t)m * N * N + r * N;
    T row[N];
#pragma unroll
    for (int j = 0; j < N; ++j) row[j] = (j <= r) ? l[j] : T(0);
    const T b = rhs[(size_t)m * N + r];

    const T xv = detail::solve_tile<T, N, TILE>(tmask, lane, row, b);

    if (store && lane < N) x[(size_t)m * N + lane] = xv;
}

template <typename T, int N>
__global__ void factor_solve_warp_per_matrix(const T* __restrict__ A,
                                             const T* __restrict__ rhs,
                                             T* __restrict__ L,
                                             T* __restrict__ x, int batch,
                                             int* __restrict__ info) {
    static_assert(N >= 2 && N <= 32);
    if (batch <= 0) return;
    constexpr int TILE = detail::tile_traits<N>::TILE;
    int m, lane;
    bool store;
    const unsigned tmask = detail::tile_setup<TILE>(batch, m, lane, store);
    const int r = min(lane, N - 1);
    const T* a = A + (size_t)m * N * N + r * N;
    T row[N];
#pragma unroll
    for (int j = 0; j < N; ++j) row[j] = a[j];

    const int fail = detail::factor_tile<T, N, TILE>(tmask, lane, row);

    const unsigned bad = __ballot_sync(tmask, fail != 0 && store);
    const int fv = __shfl_sync(tmask, fail, bad ? (__ffs(bad) - 1) : 0, TILE);
    if (store && lane == 0) info[m] = bad ? fv : 0;
    if (store && lane < N) {
        T* l = L + (size_t)m * N * N + (size_t)lane * N;
#pragma unroll
        for (int j = 0; j < N; ++j) l[j] = (j <= lane) ? row[j] : T(0);
    }
    const T b = rhs[(size_t)m * N + r];
    const T xv = detail::solve_tile<T, N, TILE>(tmask, lane, row, b);
    if (store && lane < N) x[(size_t)m * N + lane] = xv;
}

}  // namespace btl
