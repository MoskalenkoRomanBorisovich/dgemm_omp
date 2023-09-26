project(
    main
    VERSION 1.0
    LANGUAGES C
    )


add_executable(benchamrk benchamrk_liear_and_omp.c)


# ========================= OpenMP =============================
find_package(OpenMP REQUIRED)

target_link_libraries(benchamrk 
    OpenMP::OpenMP_C 
    ${BLAS_LIBRARIES})


# ========================= OpenBlas ===========================

if (NOT DEFINED BLA_VENDOR)
    find_package(BLAS REQUIRED)
    if(BLAS_FOUND)
        message("Found BLAS: ${BLAS_LIBRARIES}")
        target_link_libraries(benchamrk 
            ${BLAS_LIBRARIES})
    endif()
else()
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        target_link_libraries(benchamrk 
            ${BLAS_LIBRARIES})
    elseif()
        set(BLA_VENDOR Intel10_64ilp)
        find_package(BLAS REQUIRED)
        target_link_libraries(benchamrk_mkl 
            ${BLAS_LIBRARIES})
    endif()
endif()

# ========================= slurm run ==========================
configure_file(slurm_run.sh slurm_run.sh COPYONLY)
