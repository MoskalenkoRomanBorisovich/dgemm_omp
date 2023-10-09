#include <stdint.h>
#include <malloc.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>


void transpose(const double* a, double* b, uint_fast32_t M, uint_fast32_t N) {
    for (uint_fast32_t i = 0; i < M; i++) {
        for (uint_fast32_t j = 0; j < N; j++) {
            b[j + i * N] = a[i + j * M];
        }
    }
}

/*
  a [M][K]
  b [K][N]
  c [M][N]
*/
void blas_dgemm_simple(const uint_fast32_t M, const uint_fast32_t N, const uint_fast32_t K, const double* a, const double* b, double* c)
{
    memset(c, 0, M * N * sizeof(double));
    for (uint_fast32_t j = 0; j < N; ++j) {
        const uint_fast32_t jM = j * M;
        const uint_fast32_t jK = j * K;
        for (uint_fast32_t k = 0; k < K; ++k) {
            const uint_fast32_t kM = k * M;
            for (uint_fast32_t i = 0; i < M; ++i) {
                c[i + jM] += a[i + kM] * b[jK + k];
            }
        }
    }
}


void blas_dgemm_parallel(const uint_fast32_t M, const uint_fast32_t N, const uint_fast32_t K, const double* a, const double* b, double* c)
{
    memset(c, 0, M * N * sizeof(double));
#pragma omp parallel for schedule(static, 32)
    for (uint_fast32_t j = 0; j < N; ++j) {
        const uint_fast32_t jM = j * M;
        const uint_fast32_t jK = j * K;
        for (uint_fast32_t k = 0; k < K; ++k) {
            const uint_fast32_t kM = k * M;
            for (uint_fast32_t i = 0; i < M; ++i) {
                c[i + jM] += a[i + kM] * b[jK + k];
            }
        }
    }
}

/*
    parallel version with explicit thread scheduling
*/
void blas_dgemm_parallel_2(const uint_fast32_t M, const uint_fast32_t N, const uint_fast32_t K, const double* a, const double* b, double* c)
{
    memset(c, 0, M * N * sizeof(double));
    const uint_fast32_t num_threads = omp_get_max_threads();
    const uint_fast32_t block_size = N / num_threads;
    const uint_fast32_t block_rem = N % num_threads;
#pragma omp parallel num_threads(num_threads)
    {
        const uint_fast32_t thread_id = omp_get_thread_num();
        const uint_fast32_t block_start = thread_id * block_size + (thread_id < block_rem ? thread_id : block_rem);
        const uint_fast32_t block_end = block_start + (thread_id < block_rem ? block_size + 1 : block_size);
        for (uint_fast32_t j = block_start; j < block_end; ++j) {
            const uint_fast32_t jM = j * M;
            const uint_fast32_t jK = j * K;
            for (uint_fast32_t k = 0; k < K; ++k) {
                const uint_fast32_t kM = k * M;
                for (uint_fast32_t i = 0; i < M; ++i) {
                    c[i + jM] += a[i + kM] * b[jK + k];
                }
            }
        }
    }
}
