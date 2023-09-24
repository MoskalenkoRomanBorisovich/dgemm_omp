#include <stdint.h>
#include <malloc.h>


void transpose(const double* a, double* b, uint_fast64_t N, uint_fast64_t M) {
    for (uint_fast64_t i = 0; i < N; i++) {
        double* b_cur = &(b[i * M]);
        for (uint_fast64_t j = 0; j < M; j++) {
            b_cur[j] = a[j * N + i];
        }
    }
}

void transpose_parallel(const double* a, double* b, uint_fast64_t N, uint_fast64_t M) {
#pragma omp parallel for schedule(static, 8)
    for (uint_fast64_t i = 0; i < N; i++) {
        double* b_cur = &(b[i * M]);
        for (uint_fast64_t j = 0; j < M; j++) {
            b_cur[j] = a[j * N + i];
        }
    }
}


void blas_dgemm_simple(uint_fast64_t N, uint_fast64_t M, uint_fast64_t K, const double* a, const double* b, double* c)
{
    double* at = (double*)malloc(N * K * sizeof(double));
    transpose(a, at, N, K); // transpose to only iterate over columns for better cash use
    for (uint_fast64_t j = 0; j < M; ++j) {
        uint_fast64_t jN = j * N;
        for (uint_fast64_t i = 0; i < N; ++i) {
            const double* at_cur = &(at[i * K]);
            const double* b_cur = &(b[j * K]);
            double* c_cur = &(c[jN + i]);
            *c_cur = 0.0;
            for (uint_fast64_t k = 0; k < K; ++k) {
                *c_cur += at_cur[k] * b_cur[k];
            }
        }
    }
}


void blas_dgemm_parallel(const uint_fast64_t N, const uint_fast64_t M, const uint_fast64_t K, const double* a, const double* b, double* c)
{
    double* at = (double*)malloc(N * K * sizeof(double));
    transpose_parallel(a, at, N, K); // transpose to only iterate over columns for better cash use
#pragma omp parallel for schedule(static, 64)
    for (uint_fast64_t j = 0; j < M; ++j) {
#pragma omp parallel for schedule(static, 64)
        for (uint_fast64_t i = 0; i < N; ++i) {
            const double* at_cur = &(at[i * K]); // current column of at
            const double* b_cur = &(b[j * K]); // current column of b
            double c_cur = 0.0;
            for (uint_fast64_t k = 0; k < K; ++k) {
                c_cur += at_cur[k] * b_cur[k];
            }
            c[j * N + i] = c_cur;
        }
    }
}