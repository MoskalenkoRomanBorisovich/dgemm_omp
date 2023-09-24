project(
    test
    LANGUAGES C)

add_executable(test1 test.c)

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(test1 PUBLIC OpenMP::OpenMP_C)
endif()

add_test(
    NAME test1
    COMMAND test1)

