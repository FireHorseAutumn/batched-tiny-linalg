// Correctness tests for btl/cholesky.cuh.
// Random SPD batches (A = B B^T + n I), factor + solve via every btl
// variant, residual check against Eigen's LLT reference. Also asserts that
// a deliberately non-SPD matrix is flagged via info.
//
// Compiled as CUDA (LANGUAGE CUDA in CMake) — plain host code + launches.

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "btl/cholesky.cuh"

#define CK(x)                                                          \
    do {                                                               \
        cudaError_t e_ = (x);                                          \
        if (e_ != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s at line %d\n",              \
                    cudaGetErrorString(e_), __LINE__);                 \
            exit(1);                                                   \
        }                                                              \
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
                T acc = (r == c) ? T(n) : T(0);  // diagonal boost: modest cond
                for (int k = 0; k < n; ++k) acc += B[r * n + k] * B[c * n + k];
                Ab[r * n + c] = acc;
            }
        for (int i = 0; i < n; ++i) rhs[(size_t)b * n + i] = dist(rng);
    }
}

template <typename T>
struct Tols;
template <>
struct Tols<float> {
    static constexpr double resid = 1e-3;
    static constexpr double sol = 2e-2;
};
template <>
struct Tols<double> {
    static constexpr double resid = 1e-11;
    static constexpr double sol = 1e-7;
};

// Checks x against Eigen's LLT reference for every matrix in the batch.
// Returns max relative residual and max relative solution error.
template <typename T, int N>
void check_batch(const std::vector<T>& A, const std::vector<T>& rhs,
                 const std::vector<T>& x, int batch, double& max_resid,
                 double& max_solerr, int& n_eigen_fail) {
    max_resid = 0.0;
    max_solerr = 0.0;
    n_eigen_fail = 0;
    for (int b = 0; b < batch; ++b) {
        Eigen::Matrix<T, N, N> Am =
            Eigen::Map<const Eigen::Matrix<T, N, N>>(A.data() + (size_t)b * N * N);
        Eigen::Matrix<T, N, 1> bv =
            Eigen::Map<const Eigen::Matrix<T, N, 1>>(rhs.data() + (size_t)b * N);
        Eigen::Matrix<T, N, 1> xv =
            Eigen::Map<const Eigen::Matrix<T, N, 1>>(x.data() + (size_t)b * N);
        Eigen::LLT<Eigen::Matrix<T, N, N>> llt(Am);
        if (llt.info() != Eigen::Success) {
            ++n_eigen_fail;
            continue;
        }
        Eigen::Matrix<T, N, 1> xref = llt.solve(bv);
        double denom = (double)Am.norm() * (double)xv.norm();
        double resid = (double)(Am * xv - bv).norm() / (denom > 0 ? denom : 1);
        double solerr = (double)(xv - xref).norm() /
                        ((double)xref.norm() > 0 ? (double)xref.norm() : 1);
        if (resid > max_resid) max_resid = resid;
        if (solerr > max_solerr) max_solerr = solerr;
    }
}

template <typename T, int N>
bool run_case(int batch) {
    std::vector<T> A, rhs;
    gen_spd_batch<T>(N, batch, A, rhs, 1234 + (uint64_t)N * 7 + (uint64_t)batch);
    T *dA, *dL, *dRhs, *dX;
    int* dInfo;
    CK(cudaMalloc(&dA, A.size() * sizeof(T)));
    CK(cudaMalloc(&dL, A.size() * sizeof(T)));
    CK(cudaMalloc(&dRhs, rhs.size() * sizeof(T)));
    CK(cudaMalloc(&dX, rhs.size() * sizeof(T)));
    CK(cudaMalloc(&dInfo, batch * sizeof(int)));
    CK(cudaMemcpy(dA, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dRhs, rhs.data(), rhs.size() * sizeof(T),
                  cudaMemcpyHostToDevice));
    std::vector<T> x(rhs.size());
    std::vector<int> info(batch);
    bool ok = true;

    constexpr int TILE = (N <= 2) ? 2 : (N <= 4) ? 4 : (N <= 8) ? 8
                                     : (N <= 16) ? 16 : 32;
    const int block = 256;

    for (int variant = 0; variant < 3; ++variant) {
        const char* name;
        if (variant == 0) {
            name = "factor+solve thread/matrix";
            btl::factor_thread_per_matrix<T, N>
                <<<(batch + block - 1) / block, block>>>(dA, dL, batch, dInfo);
            btl::solve_thread_per_matrix<T, N>
                <<<(batch + block - 1) / block, block>>>(dL, dRhs, dX, batch);
        } else if (variant == 1) {
            name = "factor+solve warp/matrix";
            const int grid = (batch * TILE + block - 1) / block;
            btl::factor_warp_per_matrix<T, N><<<grid, block>>>(dA, dL, batch, dInfo);
            btl::solve_warp_per_matrix<T, N><<<grid, block>>>(dL, dRhs, dX, batch);
        } else {
            name = "factor_solve fused warp/matrix";
            const int grid = (batch * TILE + block - 1) / block;
            btl::factor_solve_warp_per_matrix<T, N>
                <<<grid, block>>>(dA, dRhs, dL, dX, batch, dInfo);
        }
        CK(cudaGetLastError());
        CK(cudaMemcpy(x.data(), dX, x.size() * sizeof(T),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(info.data(), dInfo, info.size() * sizeof(int),
                      cudaMemcpyDeviceToHost));
        int n_bad = 0;
        for (int i = 0; i < batch; ++i)
            if (info[i] != 0) ++n_bad;
        double max_resid, max_solerr;
        int n_eigen_fail;
        check_batch<T, N>(A, rhs, x, batch, max_resid, max_solerr,
                          n_eigen_fail);
        const bool pass = (n_bad == 0) && (n_eigen_fail == 0) &&
                          (max_resid < Tols<T>::resid) &&
                          (max_solerr < Tols<T>::sol);
        ok &= pass;
        printf("  n=%2d batch=%6d %-28s resid=%.3e solerr=%.3e info_bad=%d"
               " eigen_fail=%d  %s\n",
               N, batch, name, max_resid, max_solerr, n_bad, n_eigen_fail,
               pass ? "PASS" : "FAIL");
    }
    cudaFree(dA);
    cudaFree(dL);
    cudaFree(dRhs);
    cudaFree(dX);
    cudaFree(dInfo);
    return ok;
}

// Plants one non-SPD matrix (negative pivot at 0) and one that goes
// indefinite mid-factorization; both variants must flag exactly those.
template <typename T, int N>
bool run_nonspd() {
    const int batch = 8;
    std::vector<T> A, rhs;
    gen_spd_batch<T>(N, batch, A, rhs, 999 + N);
    A[(size_t)3 * N * N + 0] = T(-2);  // pivot 0 negative
    // Matrix 5: SPD until the last pivot, then indefinite.
    {
        T* M = A.data() + (size_t)5 * N * N;
        // Make the last diagonal entry exactly cancel: M[n-1][n-1] -=
        // (sum needed) is fiddly; instead subtract a large multiple of I
        // shifted to the last pivot via rank-1: M -= v v^T with v = e_0
        // scaled to kill pivot 0 is matrix 3's trick. Here: set the trailing
        // 2x2 block to be indefinite by flipping sign of the last diagonal
        // and matching off-diagonal so the matrix stays symmetric.
        T save = M[(N - 1) * N + (N - 1)];
        M[(N - 1) * N + (N - 1)] = -save - T(N);  // strongly negative
    }
    T *dA, *dL;
    int* dInfo;
    CK(cudaMalloc(&dA, A.size() * sizeof(T)));
    CK(cudaMalloc(&dL, A.size() * sizeof(T)));
    CK(cudaMalloc(&dInfo, batch * sizeof(int)));
    CK(cudaMemcpy(dA, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice));
    std::vector<int> info(batch);
    bool ok = true;

    constexpr int TILE = (N <= 2) ? 2 : (N <= 4) ? 4 : (N <= 8) ? 8
                                     : (N <= 16) ? 16 : 32;
    const int block = 256;
    for (int variant = 0; variant < 2; ++variant) {
        if (variant == 0)
            btl::factor_thread_per_matrix<T, N>
                <<<(batch + block - 1) / block, block>>>(dA, dL, batch, dInfo);
        else
            btl::factor_warp_per_matrix<T, N>
                <<<(batch * TILE + block - 1) / block, block>>>(dA, dL, batch,
                                                                dInfo);
        CK(cudaGetLastError());
        CK(cudaMemcpy(info.data(), dInfo, info.size() * sizeof(int),
                      cudaMemcpyDeviceToHost));
        bool pass = true;
        for (int b = 0; b < batch; ++b) {
            const bool expect_fail = (b == 3 || b == 5);
            const bool got_fail = info[b] != 0;
            if (expect_fail != got_fail) pass = false;
        }
        if (info[3] != 1) pass = false;  // first pivot must be the one flagged
        ok &= pass;
        printf("  n=%2d non-SPD flagging %-14s info=[%d %d %d %d %d %d %d %d]"
               "  %s\n",
               N, variant == 0 ? "thread/matrix" : "warp/matrix", info[0],
               info[1], info[2], info[3], info[4], info[5], info[6], info[7],
               pass ? "PASS" : "FAIL");
    }
    cudaFree(dA);
    cudaFree(dL);
    cudaFree(dInfo);
    return ok;
}

template <int N>
bool run_all_n() {
    bool ok = true;
    for (int batch : {1, 64, 1024, 65536}) {
        ok &= run_case<float, N>(batch);
        ok &= run_case<double, N>(batch);
    }
    ok &= run_nonspd<float, N>();
    ok &= run_nonspd<double, N>();
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= run_all_n<2>();
    ok &= run_all_n<3>();
    ok &= run_all_n<4>();
    ok &= run_all_n<6>();
    ok &= run_all_n<8>();
    ok &= run_all_n<9>();
    ok &= run_all_n<12>();
    ok &= run_all_n<16>();
    ok &= run_all_n<24>();
    ok &= run_all_n<32>();
    printf("%s\n", ok ? "ALL TESTS PASSED" : "TESTS FAILED");
    return ok ? 0 : 1;
}
