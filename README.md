# blas_dgemm with Openmp

# Run and compile
## charisma compilation
From project directory call following commands:
    module load cmake/3.21.3 
    module load INTEL/oneAPI_2022_env
    module load OpenBlas/v0.3.18

    mkdir build
    cd build
    cmake ../
    cmake --build . --config Release --target all

## run slurm
From project directory call following commands:
    cd build/apps
    bash run_sbatches.sh

Output will be in blas_dgemm_{n_cores}.log files

## Benchmark results
Benchmarking results from charisma are available in ./benchmark_results/speed_analysis.ipynb