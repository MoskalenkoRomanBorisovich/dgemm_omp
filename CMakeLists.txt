cmake_minimum_required(VERSION 3.2)

project(
    my_dgemm
    VERSION 1.0
    LANGUAGES C
    )
add_subdirectory(apps)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
elseif(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

enable_testing ()
add_subdirectory(test)