add_executable(tests_stdexec_nvcc_failure)
target_sources(tests_stdexec_nvcc_failure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_failure.cpp)

target_link_libraries(tests_stdexec_nvcc_failure PRIVATE stdexec::stdexec)

# If the Kokkos CXX compiler ID is NVIDIA, remove the target from the default/all targets.
# Add a test that will try to compile the target, but will fail.
if(Kokkos_CXX_COMPILER_ID STREQUAL NVIDIA)

    set_target_properties(
        tests_stdexec_nvcc_failure
        PROPERTIES
            EXCLUDE_FROM_ALL TRUE
            EXCLUDE_FROM_DEFAULT_BUILD TRUE
    )

    add_test(
        NAME tests_stdexec_nvcc_failure
        COMMAND ${CMAKE_COMMAND}
                --build .
                --config $<CONFIGURATION>
                --target tests_stdexec_nvcc_failure
                --verbose
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    set_tests_properties(
        tests_stdexec_nvcc_failure
        PROPERTIES
            PASS_REGULAR_EXPRESSION "include/stdexec/__detail/__meta\.hpp\\([0-9]+\\): error: no instance of constructor \"stdexec::__mstring\" matches the argument list"
    )
endif()
