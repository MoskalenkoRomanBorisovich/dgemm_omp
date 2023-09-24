project(
    main
    VERSION 1.0
    LANGUAGES C
    )


# ========================= OpenBlas ===========================
add_executable(benchamrk_openblas benchamrk_liear_and_omp.c)

set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("BALS PATH: ${BLAS_LIBRARIES}")
    target_link_libraries(benchamrk_openblas 
        ${BLAS_LIBRARIES})
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(benchamrk_openblas 
    OpenMP::OpenMP_C 
    ${BLAS_LIBRARIES})



# ========================= MKL ================================
add_executable(benchamrk_mkl benchamrk_liear_and_omp.c)

set(BLA_VENDOR Intel10_64ilp)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("BALS PATH: ${BLAS_LIBRARIES}")
    target_link_libraries(benchamrk_mkl 
        ${BLAS_LIBRARIES})
endif()

target_link_libraries(benchamrk_mkl 
    OpenMP::OpenMP_C 
    ${BLAS_LIBRARIES})

# ========================= slurm run ==========================
configure_file(slurm_run.sh slurm_run.sh COPYONLY)
