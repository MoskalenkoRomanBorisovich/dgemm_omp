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
benchmark given matrix multiplication function on random matrixes
*/

#define TOL 1e-6

typedef void (*blas_dgemm_t)(const uint_fast32_t M, const uint_fast32_t N, const uint_fast32_t K, const double* a, const double* b, double* c);

void benchmark_func(
    blas_dgemm_t* f_arr,
    const uint_fast8_t n_funcs,
    const uint_fast32_t N,
    const uint_fast32_t n_runs,
    const unsigned int seed,
    double* t_sec,
    double* flops)
{
    srand(seed);
    const uint_fast32_t N2 = N * N;
    double* a = (double*)malloc(N2 * sizeof(double));
    double* b = (double*)malloc(N2 * sizeof(double));
    double* c_arr = (double*)malloc(n_funcs * N2 * sizeof(double)); // array of result matrixes
    double start, finish;
    memset(t_sec, 0, n_funcs * sizeof(double));
    memset(flops, 0, n_funcs * sizeof(double));
    for (uint_fast32_t i = 0; i < n_runs; ++i) {
        for (uint_fast32_t i = 0; i < N2; ++i)
            a[i] = (double)rand() / RAND_MAX;
        for (uint_fast32_t i = 0; i < N2; ++i)
            b[i] = (double)rand() / RAND_MAX;

        for (uint_fast8_t j = 0; j < n_funcs; ++j) { // run all functions on random matrix
            start = omp_get_wtime();
            (*(f_arr[j]))(N, N, N, a, b, &c_arr[j * N2]);
            finish = omp_get_wtime();
            t_sec[j] += (finish - start);
        }

        for (uint_fast8_t j = 0; j < n_funcs - 1; ++j) {
            double dif = mat_diff(&c_arr[j * N2], &c_arr[(j + 1) * N2], N, N);
            if (dif > TOL) {
                printf("Matrixes are not equal for %d and %d, difference: %lf\n", j, j + 1, dif);
                fflush(stdout);
                assert(0);
            }
        }
    }

    for (uint_fast8_t j = 0; j < n_funcs; ++j) {
        t_sec[j] /= n_runs;
        flops[j] = 1e-9 * get_flop_count(N, N, N) / (t_sec[j]);
    }
    free(a);
    free(b);
    free(c_arr);
}


void cblas_wrapper(const uint_fast32_t M, const uint_fast32_t N, const uint_fast32_t K, const double* a, const double* b, double* c)
{
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, a, M, b, K, 0.0, c, M);
}

void benchmark_all(
    unsigned int seed,
    uint_fast32_t n_runs,
    uint_fast32_t N)
{
    const uint8_t n_functions = 4;
    double t_sec[n_functions];
    double flops[n_functions];
    blas_dgemm_t functions[] = { cblas_wrapper, blas_dgemm_parallel, blas_dgemm_parallel_2, blas_dgemm_simple };

    benchmark_func(functions, n_functions, N, n_runs, seed, t_sec, flops);
    for (uint_fast8_t i = 0; i < n_functions; ++i) {
        printf("%f %15f\n", t_sec[i], flops[i]);
    }
}

/*
main function
params:
    1. matrix size
    2. number of runs for each function
    3. seed for matrix generation
*/
int main(int argc, char** argv)
{
    uint_fast32_t N = 500;
    uint_fast32_t n_runs = 10;
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