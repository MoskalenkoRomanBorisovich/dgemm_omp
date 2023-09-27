#! /bin/bash

for n_cores in 1 2 4 8 16; do
    sbatch --cpus-per-task=$n_cores --job-name="blas_dgemm_$n_cores" --output="blas_dgemm_$n_cores.log" slurm_run.sh
done
