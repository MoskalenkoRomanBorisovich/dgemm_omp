project(
    main
    VERSION 1.0
    LANGUAGES C
    )




# ========================= OpenMP =============================
find_package(OpenMP REQUIRED)


# ========================= Custom BLAS vendor =============================

if (DEFINED BLA_VENDOR)
    add_executable(benchamrk_blas_custom benchamrk_liear_and_omp.c)
    message("Searching for blas vendor: ${BLA_VENDOR}")
    find_package(BLAS REQUIRED)
    message("Found BLAS: ${BLAS_LIBRARIES}")
    target_link_libraries(benchamrk_blas_custom 
        ${BLAS_LIBRARIES} OpenMP::OpenMP_C)
endif()

# ========================= OpenBlas ===========================


set(BLA_VENDOR OpenBLAS)
find_package(BLAS)
if(BLAS_FOUND)
    add_executable(benchamrk_openblas benchamrk_liear_and_omp.c)
    message("Found BLAS: ${BLAS_LIBRARIES}")
    target_link_libraries(benchamrk_openblas 
        ${BLAS_LIBRARIES} OpenMP::OpenMP_C)
endif()

# ========================= MKL ===========================

set(BLA_VENDOR Intel10_64ilp)
find_package(BLAS)
if(BLAS_FOUND)
    add_executable(benchamrk_mkl benchamrk_liear_and_omp.c)
    message("Found BLAS: ${BLAS_LIBRARIES}")
    target_link_libraries(benchamrk_mkl 
        ${BLAS_LIBRARIES} OpenMP::OpenMP_C)
endif()

# ========================= slurm run ==========================
configure_file(slurm_run.sh slurm_run.sh COPYONLY)
configure_file(run_sbatches.sh run_sbatches.sh COPYONLY)
