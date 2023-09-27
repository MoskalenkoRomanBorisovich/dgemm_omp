#! /bin/bash
# TODO

#SBATCH --time=00:10:00
#SBATCH --constraint="type_d"

N_RUNS=10
SEED=12345
ls
for matrix_size in 500 1000 1500; do
    echo "Openblas"
    ./benchamrk_openblas $matrix_size $N_RUNS $SEED
    echo "MKL"
    ./benchamrk_mkl $matrix_size $N_RUNS $SEED

done
