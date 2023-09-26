cmake_minimum_required(VERSION 3.2)

project(
    my_dgemm
    VERSION 1.0
    LANGUAGES C
    )
add_subdirectory(apps)

enable_testing ()
add_subdirectory(test)