# Helper for adding one benchmark.
function(add_one_benchmark)

    cmake_parse_arguments(
        aob_args
        ""
        "NAME"
        "ADDITIONAL_LINK_LIBRARIES"
        ${ARGN}
    )

    # Set executable name and primary source file.
    cmake_path(SET SOURCE_FILE "benchmark_${aob_args_NAME}.cpp")
    cmake_path(ABSOLUTE_PATH SOURCE_FILE)

    get_parent_relpath_repr(DIRECTORY "${PROJECT_SOURCE_DIR}" FILENAME "${SOURCE_FILE}")

    set(EXECUTABLE_NAME "${gprr_PARENT_RELPATH_REPR}_${aob_args_NAME}")

    # Add source file to the documentation list.
    set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${SOURCE_FILE})

    # Create benchmark executable.
    add_executable(${EXECUTABLE_NAME})

    # Add the source file to the executable.
    target_sources(
        ${EXECUTABLE_NAME}
        PRIVATE
            ${SOURCE_FILE}
    )

    # Add current binary directory to the compile definitions to allow benchmarks to output files.
    target_compile_definitions(
        ${EXECUTABLE_NAME}
        PRIVATE
            CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Link the executable to Google Benchmark and our custom main.
    target_link_libraries(
        ${EXECUTABLE_NAME}
        PRIVATE
            benchmark_custom_main
            plog
    )

    # Additional link libraries
    if(DEFINED aob_args_ADDITIONAL_LINK_LIBRARIES)
        target_link_libraries(
            ${EXECUTABLE_NAME}
            PRIVATE
               ${aob_args_ADDITIONAL_LINK_LIBRARIES}
        )
    endif()

    # Allow benchmarks to reference files in this repository directly.
    target_include_directories(
        ${EXECUTABLE_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}
    )

    # There must be a Python script to drive the benchmark.
    cmake_path(SET SCRIPT_FILE "benchmark_${aob_args_NAME}.py")
    cmake_path(ABSOLUTE_PATH SCRIPT_FILE)

    # Add the benchmark to CTest.
    add_test(
        NAME ${EXECUTABLE_NAME}
        COMMAND python3 ${SCRIPT_FILE}
                --target $<TARGET_FILE:${EXECUTABLE_NAME}>
                --
                #--benchmark_min_time=1x
    )
    set_property(
        TEST ${EXECUTABLE_NAME}
        APPEND PROPERTY
            ENVIRONMENT_MODIFICATION PYTHONPATH=path_list_prepend:${CMAKE_SOURCE_DIR}
    )

endfunction()

#---- Find Benchmark.
string(JSON Benchmark_REQUIRED_VERSION GET "${VERSION_JSON}" dependencies googlebenchmark value)

find_package(
    benchmark
    ${Benchmark_REQUIRED_VERSION}
    CONFIG
    REQUIRED
)

# Add headers to documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

#---- Make an object library containing our custom main.
add_library(benchmark_custom_main OBJECT)
target_sources(benchmark_custom_main PRIVATE main.cpp)
target_link_libraries(
    benchmark_custom_main
    PUBLIC
        benchmark::benchmark
    PRIVATE
        Kokkos::kokkoscore
)

add_subdirectory(cg)
add_subdirectory(graph)
add_subdirectory(newton)
add_subdirectory(pcg)
