#include <stdio.h>
#include <malloc.h>
#include <memory.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>


#define TOL 1e-6 // tolerance for matrix comparison

void print_matrix(double* a, uint_fast64_t n, uint_fast64_t m)
{
    for (uint_fast64_t i = 0; i < n; i++) {
        for (uint_fast64_t j = 0; j < m; j++) {
            printf("%f ", a[j * n + i]);
        }
        printf("\n");
    }
}


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

/*
number of operations for matrix multiplication
*/
uint_fast64_t get_flop_count(uint_fast64_t N, uint_fast64_t M, uint_fast64_t K) {
    return 2 * N * M * K;
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
#pragma omp parallel for
    for (uint_fast64_t j = 0; j < M; ++j) {
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

/*
squared Frobenius norm of matrix difference
*/
double mat_diff(const double* a, const double* b, uint_fast64_t N, uint_fast64_t M) {
    double res = 0.0;
    double d;
    for (uint_fast64_t i = 0, i_end = N * M; i < i_end; ++i) {
        d = a[i] - b[i];
        res += d * d;
    }
    return res;
}

/*
test single thread algorithm
*/
void test1_single() {
    const uint_fast64_t N = 2;
    const uint_fast64_t K = 3;
    const uint_fast64_t M = 3;
    double a[] = { 1, 2, 3, 4, 5, 6 };
    double b[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    double c[N * M];

    blas_dgemm_simple(N, M, K, a, b, c);

    double c_correct[] = { 22, 28, 49, 64, 76, 100 };

    assert(mat_diff(c, c_correct, N, M) < TOL);
}

/*
compare single thread and parallel algorithm
*/
void test2_single_to_parallel() {
    const uint_fast64_t N = 32;
    const uint_fast64_t K = 64;
    const uint_fast64_t M = 64;
    double a[N * K];
    double b[K * M];
    double c_s[M * N];
    double c_p[M * N];
    const uint_fast64_t n_runs = 10;


    for (uint_fast64_t i = 0; i < n_runs; ++i) {
        for (uint_fast64_t i = 0; i < N * K; ++i)
            a[i] = (double)rand() / RAND_MAX;


        for (uint_fast64_t i = 0; i < K * M; ++i)
            b[i] = (double)rand() / RAND_MAX;

        blas_dgemm_parallel(N, M, K, a, b, c_p);
        blas_dgemm_simple(N, M, K, a, b, c_s);
        assert(mat_diff(c_p, c_s, N, M) < TOL);
    }
}

/*
benchmark given matrix multiplication function on random matrices
*/
void benchmark_func(
    void (*f)(const uint_fast64_t N, const uint_fast64_t M, const uint_fast64_t K, const double* a, const double* b, double* c),
    const uint_fast64_t N,
    const uint_fast64_t n_runs,
    const unsigned int seed,
    double* t_sec,
    double* flops)
{
    srand(seed);
    const uint_fast64_t N2 = N * N;
    double* a = (double*)malloc(N2 * sizeof(double));
    double* b = (double*)malloc(N2 * sizeof(double));
    double* c = (double*)malloc(N2 * sizeof(double));
    double start, finish;
    *t_sec = 0.0;
    for (uint_fast64_t i = 0; i < n_runs; ++i) {
        for (uint_fast64_t i = 0; i < N2; ++i)
            a[i] = (double)rand() / RAND_MAX;
        for (uint_fast64_t i = 0; i < N2; ++i)
            b[i] = (double)rand() / RAND_MAX;

        start = omp_get_wtime();
        (*f)(N, N, N, a, b, c);
        finish = omp_get_wtime();
        *t_sec += (finish - start);
    }
    *t_sec /= n_runs;
    *flops = 1e-9 * get_flop_count(N, N, N) / (*t_sec);

    free(a);
    free(b);
    free(c);
}


void benchmark_all(
    unsigned int seed,
    uint_fast64_t n_runs,
    uint_fast64_t N)
{
    double t_sec;
    double flops;

    benchmark_func(blas_dgemm_simple, N, n_runs, seed, &t_sec, &flops);
    printf("%f %15f\n", t_sec, flops);
    benchmark_func(blas_dgemm_parallel, N, n_runs, seed, &t_sec, &flops);
    printf("%f %15f\n", t_sec, flops);
}

int main(int argc, char** argv)
{
    test1_single();
    test2_single_to_parallel();

    uint_fast64_t N = 500;
    uint_fast64_t n_runs = 10;
    unsigned int seed = 123;

    switch (argc)
    {
    case 4:
        seed = atol(argv[3]);
    case 3:
        n_runs = atol(argv[2]);
    case 2:
        N = atol(argv[1]);
        break;
    default:
        break;
    }
    printf("%lu %lu %d\n", N, n_runs, seed);
    fflush(stdout);
    benchmark_all(seed, n_runs, N);

    return 0;
}