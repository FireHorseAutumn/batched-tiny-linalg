// Benchmark: btl Cholesky (thread-per-matrix, warp-per-matrix) vs
// cuSOLVER batched (potrfBatched + potrsBatched) vs Eigen CPU LLT.
// Median-of-N timing with warmup; results appended to JSON.
//
// Fairness notes:
//  - cuSOLVER's potrf is in-place, so A is restored from a pristine copy
//    before every iteration; the restore copy is OUTSIDE the timed region
//    (biased slightly in cuSOLVER's favor — conservative for the btl claim).
//  - The cuSOLVER pointer-array setup is likewise outside the timed region.
//  - Eigen uses compile-time fixed-size matrices (no allocator overhead) —
//    the flattering-to-Eigen baseline.

#include <Eigen/Dense>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "btl/cholesky.cuh"

#define CK(x)                                                           \
    do {                                                                \
        cudaError_t e_ = (x);                                           \
        if (e_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at line %d\n",               \
                    cudaGetErrorString(e_), __LINE__);                  \
            exit(1);                                                    \
        }                                                               \
    } while (0)

#define CKS(x)                                                          \
    do {                                                                \
        cusolverStatus_t s_ = (x);                                      \
        if (s_ != CUSOLVER_STATUS_SUCCESS) {                            \
            fprintf(stderr, "cuSOLVER error %d at line %d\n", (int)s_,  \
                    __LINE__);                                          \
            exit(1);                                                    \
        }                                                               \
    } while (0)

namespace {

template <typename T>
void gen_spd_batch(int n, int batch, std::vector<T>& A, std::vector<T>& rhs,
                   uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    A.assign((size_t)batch * n * n, T(0));
    rhs.resize((size_t)batch * n);
    std::vector<T> B(n * n);
    for (int b = 0; b < batch; ++b) {
        for (auto& v : B) v = dist(rng);
        T* Ab = A.data() + (size_t)b * n * n;
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) {
                T acc = (r == c) ? T(n) : T(0);
                for (int k = 0; k < n; ++k) acc += B[r * n + k] * B[c * n + k];
                Ab[r * n + c] = acc;
            }
        for (int i = 0; i < n; ++i) rhs[(size_t)b * n + i] = dist(rng);
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Median GPU time (ms) of op(), timed with CUDA events, warmup first.
template <typename F>
double time_gpu_ms(F&& op, int warmup, int iters) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s));
    CK(cudaEventCreate(&e));
    for (int i = 0; i < warmup; ++i) op();
    CK(cudaDeviceSynchronize());
    std::vector<double> ts;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(s));
        op();
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        float ms;
        CK(cudaEventElapsedTime(&ms, s, e));
        ts.push_back(ms);
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return median(std::move(ts));
}

template <typename F>
double time_cpu_ms(F&& op, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) op();
    std::vector<double> ts;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        op();
        const auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median(std::move(ts));
}

// cuSOLVER 13.2 declares the batched APIs with non-const pointer arrays.
template <typename T>
cusolverStatus_t potrf_batched(cusolverDnHandle_t h, cublasFillMode_t uplo,
                               int n, T* A[], int lda, int* info, int batch);
template <>
cusolverStatus_t potrf_batched<float>(cusolverDnHandle_t h,
                                      cublasFillMode_t uplo, int n, float* A[],
                                      int lda, int* info, int batch) {
    return cusolverDnSpotrfBatched(h, uplo, n, A, lda, info, batch);
}
template <>
cusolverStatus_t potrf_batched<double>(cusolverDnHandle_t h,
                                       cublasFillMode_t uplo, int n,
                                       double* A[], int lda, int* info,
                                       int batch) {
    return cusolverDnDpotrfBatched(h, uplo, n, A, lda, info, batch);
}

template <typename T>
cusolverStatus_t potrs_batched(cusolverDnHandle_t h, cublasFillMode_t uplo,
                               int n, int nrhs, T* A[], int lda, T* B[],
                               int ldb, int* info, int batch);
template <>
cusolverStatus_t potrs_batched<float>(cusolverDnHandle_t h,
                                      cublasFillMode_t uplo, int n, int nrhs,
                                      float* A[], int lda, float* B[], int ldb,
                                      int* info, int batch) {
    return cusolverDnSpotrsBatched(h, uplo, n, nrhs, A, lda, B, ldb, info,
                                   batch);
}
template <>
cusolverStatus_t potrs_batched<double>(cusolverDnHandle_t h,
                                       cublasFillMode_t uplo, int n, int nrhs,
                                       double* A[], int lda, double* B[],
                                       int ldb, int* info, int batch) {
    return cusolverDnDpotrsBatched(h, uplo, n, nrhs, A, lda, B, ldb, info,
                                   batch);
}

template <typename T, int N>
nlohmann::json run_config(int batch, cusolverDnHandle_t solver) {
    constexpr int TILE = (N <= 2) ? 2 : (N <= 4) ? 4 : (N <= 8) ? 8
                                     : (N <= 16) ? 16 : 32;
    constexpr int block = 256;
    const size_t nmat = (size_t)batch * N * N;
    const size_t nvec = (size_t)batch * N;

    std::vector<T> A, rhs;
    gen_spd_batch<T>(N, batch, A, rhs, 42 + (uint64_t)N * 3 + batch);

    T *dA, *dAw, *dRhs, *dBw, *dL, *dX;
    int* dInfo;
    CK(cudaMalloc(&dA, nmat * sizeof(T)));
    CK(cudaMalloc(&dAw, nmat * sizeof(T)));
    CK(cudaMalloc(&dRhs, nvec * sizeof(T)));
    CK(cudaMalloc(&dBw, nvec * sizeof(T)));
    CK(cudaMalloc(&dL, nmat * sizeof(T)));
    CK(cudaMalloc(&dX, nvec * sizeof(T)));
    CK(cudaMalloc(&dInfo, batch * sizeof(int)));
    CK(cudaMemcpy(dA, A.data(), nmat * sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dRhs, rhs.data(), nvec * sizeof(T), cudaMemcpyHostToDevice));

    // cuSOLVER pointer arrays (setup outside timed region).
    std::vector<T*> hA(batch), hB(batch);
    for (int b = 0; b < batch; ++b) {
        hA[b] = dAw + (size_t)b * N * N;
        hB[b] = dBw + (size_t)b * N;
    }
    T **dAp, **dBp;
    CK(cudaMalloc(&dAp, batch * sizeof(T*)));
    CK(cudaMalloc(&dBp, batch * sizeof(T*)));
    CK(cudaMemcpy(dAp, hA.data(), batch * sizeof(T*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBp, hB.data(), batch * sizeof(T*), cudaMemcpyHostToDevice));

    const int gpu_warmup = 5, gpu_iters = 20;
    const int cpu_warmup = 1, cpu_iters = batch >= 8192 ? 5 : 11;

    // --- btl thread-per-matrix (fused factor+solve) ---
    const int tgrid = (batch + block - 1) / block;
    const double btl_thread = time_gpu_ms(
        [&] {
            btl::factor_solve_thread_per_matrix<T, N>
                <<<tgrid, block>>>(dA, dRhs, dL, dX, batch, dInfo);
        },
        gpu_warmup, gpu_iters);

    // --- btl warp-per-matrix (fused factor+solve) ---
    const int wgrid = (batch * TILE + block - 1) / block;
    const double btl_warp = time_gpu_ms(
        [&] {
            btl::factor_solve_warp_per_matrix<T, N>
                <<<wgrid, block>>>(dA, dRhs, dL, dX, batch, dInfo);
        },
        gpu_warmup, gpu_iters);

    // --- cuSOLVER batched potrf + potrs ---
    auto restore = [&] {
        CK(cudaMemcpy(dAw, dA, nmat * sizeof(T), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(dBw, dRhs, nvec * sizeof(T), cudaMemcpyDeviceToDevice));
    };
    auto cusolver_op = [&] {
        CKS(potrf_batched<T>(solver, CUBLAS_FILL_MODE_LOWER, N, dAp, N, dInfo,
                             batch));
        CKS(potrs_batched<T>(solver, CUBLAS_FILL_MODE_LOWER, N, 1, dAp, N,
                             dBp, N, dInfo, batch));
    };
    // cuSOLVER needs the A/rhs restore between iterations, outside the timed
    // region — time it manually rather than through time_gpu_ms.
    double cusolver_med;
    {
        cudaEvent_t s, e;
        CK(cudaEventCreate(&s));
        CK(cudaEventCreate(&e));
        for (int i = 0; i < gpu_warmup; ++i) {
            restore();
            cusolver_op();
        }
        CK(cudaDeviceSynchronize());
        std::vector<double> ts;
        for (int i = 0; i < gpu_iters; ++i) {
            restore();
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(s));
            cusolver_op();
            CK(cudaEventRecord(e));
            CK(cudaEventSynchronize(e));
            float ms;
            CK(cudaEventElapsedTime(&ms, s, e));
            ts.push_back(ms);
        }
        cusolver_med = median(std::move(ts));
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }

    // --- Eigen CPU (fixed-size LLT + solve per matrix) ---
    // x is written back and checksummed so the loop cannot be eliminated.
    std::vector<T> x_cpu(nvec);
    volatile double sink = 0;
    const double eigen_ms = time_cpu_ms(
        [&] {
            for (int b = 0; b < batch; ++b) {
                Eigen::Matrix<T, N, N> Am =
                    Eigen::Map<const Eigen::Matrix<T, N, N>>(
                        A.data() + (size_t)b * N * N);
                Eigen::Matrix<T, N, 1> bv =
                    Eigen::Map<const Eigen::Matrix<T, N, 1>>(
                        rhs.data() + (size_t)b * N);
                Eigen::LLT<Eigen::Matrix<T, N, N>> llt(Am);
                Eigen::Map<Eigen::Matrix<T, N, 1>>(x_cpu.data() +
                                                   (size_t)b * N) =
                    llt.solve(bv);
            }
            sink = sink + (double)x_cpu[(size_t)batch / 2 * N];
        },
        cpu_warmup, cpu_iters);
    if (sink == 42.42) printf("impossible\n");  // keep the sink observable

    cudaFree(dA);
    cudaFree(dAw);
    cudaFree(dRhs);
    cudaFree(dBw);
    cudaFree(dL);
    cudaFree(dX);
    cudaFree(dInfo);
    cudaFree(dAp);
    cudaFree(dBp);

    const double best_btl = std::min(btl_thread, btl_warp);
    const char* winner = "btl";
    double best = best_btl;
    if (cusolver_med < best) { best = cusolver_med; winner = "cusolver"; }
    if (eigen_ms < best) { best = eigen_ms; winner = "eigen_cpu"; }

    nlohmann::json j;
    j["n"] = N;
    j["batch"] = batch;
    j["dtype"] = std::is_same_v<T, float> ? "float32" : "float64";
    j["ms"] = {{"btl_thread_per_matrix", btl_thread},
               {"btl_warp_per_matrix", btl_warp},
               {"cusolver_batched_potrf_potrs", cusolver_med},
               {"eigen_cpu_llt", eigen_ms}};
    j["speedup_vs_cusolver"] = cusolver_med / best_btl;
    j["winner"] = winner;
    // Kill-criterion flag: cuSOLVER wins at n <= 32 with batch >= 64.
    j["kill_criterion_hit"] =
        (cusolver_med < best_btl) && batch >= 64;
    printf("  n=%2d batch=%6d %-7s btl_thread=%9.4f btl_warp=%9.4f"
           " cusolver=%9.4f eigen_cpu=%9.4f ms  winner=%s (%.2fx vs cuSOLVER)\n",
           N, batch, std::is_same_v<T, float> ? "f32" : "f64", btl_thread,
           btl_warp, cusolver_med, eigen_ms, winner,
           cusolver_med / best_btl);
    return j;
}

template <int N>
void run_n(const std::vector<int>& batches, cusolverDnHandle_t solver,
           nlohmann::json& configs) {
    for (int batch : batches) {
        configs.push_back(run_config<float, N>(batch, solver));
        configs.push_back(run_config<double, N>(batch, solver));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* out_path =
        argc > 1 ? argv[1] : "bench_cholesky_results.json";
    const std::vector<int> batches = {64, 1024, 65536};

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    cusolverDnHandle_t solver;
    CKS(cusolverDnCreate(&solver));

    nlohmann::json configs = nlohmann::json::array();
    run_n<2>(batches, solver, configs);
    run_n<3>(batches, solver, configs);
    run_n<4>(batches, solver, configs);
    run_n<6>(batches, solver, configs);
    run_n<8>(batches, solver, configs);
    run_n<9>(batches, solver, configs);
    run_n<12>(batches, solver, configs);
    run_n<16>(batches, solver, configs);
    run_n<24>(batches, solver, configs);
    run_n<32>(batches, solver, configs);

    nlohmann::json out;
    out["machine"] = {
        {"gpu", prop.name},
        {"compute_capability",
         std::to_string(prop.major) + "." + std::to_string(prop.minor)},
        {"cuda_runtime_version", CUDART_VERSION},
        {"eigen", "system Eigen3 (fixed-size LLT, single thread)"},
        {"timing", "median of 20 (GPU, CUDA events) / 5-11 (CPU), warmup first"}};
    out["notes"] =
        "cuSOLVER potrf is in-place: A/rhs restores are outside the timed "
        "region (generous to cuSOLVER). btl time is the fused factor_solve "
        "kernel, A read-only.";
    out["configs"] = configs;

    int kills = 0;
    for (const auto& c : configs)
        if (c["kill_criterion_hit"].get<bool>()) ++kills;
    out["kill_criterion_hits"] = kills;

    FILE* f = fopen(out_path, "w");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    const std::string s = out.dump(2);
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
    printf("wrote %s (%d configs, %d kill-criterion hits)\n", out_path,
           (int)configs.size(), kills);
    CKS(cusolverDnDestroy(solver));
    return 0;
}
