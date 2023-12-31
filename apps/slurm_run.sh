#! /bin/bash

#SBATCH --time=01:00:00
#SBATCH --constraint="type_d"

N_RUNS=10
SEED=12345
for matrix_size in 500 1000 1500; do
    echo "Openblas"
    ./benchamrk_openblas $matrix_size $N_RUNS $SEED
    echo "MKL"
    ./benchamrk_mkl $matrix_size $N_RUNS $SEED

done
