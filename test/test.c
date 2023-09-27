#include <stdlib.h>
#include <assert.h>
#include "../source/my_dgemm.h"
#include "../source/utils.h"

#define TOL 1e-6 // tolerance for matrix comparison

/*
test single thread algorithm
*/
void test1_single() {
    const uint_fast32_t N = 2;
    const uint_fast32_t K = 3;
    const uint_fast32_t M = 3;
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
    const uint_fast32_t N = 32;
    const uint_fast32_t K = 64;
    const uint_fast32_t M = 64;
    double a[N * K];
    double b[K * M];
    double c_s[M * N];
    double c_p[M * N];
    const uint_fast32_t n_runs = 10;

    printf("number of threads: %d\n", omp_get_max_threads());
    for (uint_fast32_t i = 0; i < n_runs; ++i) {
        for (uint_fast32_t i = 0; i < N * K; ++i)
            a[i] = (double)rand() / RAND_MAX;


        for (uint_fast32_t i = 0; i < K * M; ++i)
            b[i] = (double)rand() / RAND_MAX;

        blas_dgemm_parallel(N, M, K, a, b, c_p);
        blas_dgemm_simple(N, M, K, a, b, c_s);
        assert(mat_diff(c_p, c_s, N, M) < TOL);
    }
}



int main()
{
    test1_single();
    test2_single_to_parallel();
}