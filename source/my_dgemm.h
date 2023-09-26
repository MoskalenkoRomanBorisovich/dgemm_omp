#include <stdint.h>
#include <malloc.h>


void transpose(const double* a, double* b, uint_fast32_t N, uint_fast32_t M) {
    for (uint_fast32_t i = 0; i < N; i++) {
        double* b_cur = &(b[i * M]);
        for (uint_fast32_t j = 0; j < M; j++) {
            b_cur[j] = a[j * N + i];
        }
    }
}

void transpose_parallel(const double* a, double* b, uint_fast32_t N, uint_fast32_t M) {
#pragma omp parallel for schedule(static, 64)
    for (uint_fast32_t i = 0; i < N; i++) {
        double* b_cur = &(b[i * M]);
#pragma omp ismd
        for (uint_fast32_t j = 0; j < M; j++) {
            b_cur[j] = a[j * N + i];
        }
    }
}


void blas_dgemm_simple(uint_fast32_t N, uint_fast32_t M, uint_fast32_t K, const double* a, const double* b, double* c)
{
    double* at = (double*)malloc(N * K * sizeof(double));
    transpose(a, at, N, K); // transpose to only iterate over columns for better cash use
    for (uint_fast32_t j = 0; j < M; ++j) {
        uint_fast32_t jN = j * N;
        for (uint_fast32_t i = 0; i < N; ++i) {
            const double* at_cur = &(at[i * K]);
            const double* b_cur = &(b[j * K]);
            double* c_cur = &(c[jN + i]);
            *c_cur = 0.0;
            for (uint_fast32_t k = 0; k < K; ++k) {
                *c_cur += at_cur[k] * b_cur[k];
            }
        }
    }
    free(at);
}


void blas_dgemm_parallel(const uint_fast32_t N, const uint_fast32_t M, const uint_fast32_t K, const double* a, const double* b, double* c)
{
    double* at = (double*)malloc(N * K * sizeof(double));
    transpose_parallel(a, at, N, K); // transpose to only iterate over columns for better cash use
#pragma omp parallel for schedule(auto)
    for (uint_fast32_t j = 0; j < M; ++j) {
        double* c_col = &(c[j * N]);
        const double* b_cur = &(b[j * K]); // current column of b
        for (uint_fast32_t i = 0; i < N; ++i) {
            const double* at_cur = &(at[i * K]); // current column of at
            double c_cur = 0.0;
#pragma omp simd reduction(+:c_cur)
            for (uint_fast32_t k = 0; k < K; ++k) {
                c_cur += at_cur[k] * b_cur[k];
            }
            c_col[i] = c_cur;
        }
    }
    free(at);
}