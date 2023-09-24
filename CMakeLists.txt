cmake_minimum_required(VERSION 3.22)

project(
    main
    VERSION 1.0
    LANGUAGES C
    )
add_executable(main main.c)

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_C)
endif()

# set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
if(BLAS_C_FOUND)
    target_link_libraries(main PUBLIC BLAS::BLAS_C)
endif()