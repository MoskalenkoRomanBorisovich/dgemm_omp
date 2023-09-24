#include <stdio.h>
#include <malloc.h>
#include <memory.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>


#include <cblas.h>
#include "../source/my_dgemm.h"
#include "../source/utils.h"

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


void cblas_wrapper(const uint_fast64_t N, const uint_fast64_t M, const uint_fast64_t K, const double* a, const double* b, double* c)
{
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1.0, a, N, b, N, 0.0, c, N);
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
    benchmark_func(cblas_wrapper, N, n_runs, seed, &t_sec, &flops);
    printf("%f %15f\n", t_sec, flops);
    benchmark_func(blas_dgemm_parallel, N, n_runs, seed, &t_sec, &flops);
    printf("%f %15f\n", t_sec, flops);
}

int main(int argc, char** argv)
{
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