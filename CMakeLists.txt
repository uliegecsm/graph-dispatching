cmake_minimum_required(VERSION 3.25)

#---- Read the version file.
file(READ version.json VERSION_JSON)

#---- Get the version of graph-dispatching.
string(JSON GRAPH_DISPATCHING_VERSION GET "${VERSION_JSON}" "graph-dispatching" "value")

#---- Define the project. It uses C++ only.
project(
    graph-dispatching
    VERSION ${GRAPH_DISPATCHING_VERSION}
    LANGUAGES CXX
)

#---- Enable HPX testing.
option(ENABLE_HPX_TESTING "Enable testing of HPX adaptors." ON)

#---- Enable clang-tidy. This might bloat the compile time.
option(ENABLE_CLANG_TIDY OFF)

#---- Enable further debugging by exporting compile commands.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#---- Building Doxygen documentation is OFF by default.
option(ENABLE_DOXYGEN "Enable building Doxygen documentation" OFF)

#---- Enable multi-GPU support (tests, ...).
if(DEFINED ENV{GRAPH_DISPATCHING_ENABLE_MULTIGPU})
    set(GRAPH_DISPATCHING_ENABLE_MULTIGPU_DEFAULT $ENV{GRAPH_DISPATCHING_ENABLE_MULTIGPU})
else()
    set(GRAPH_DISPATCHING_ENABLE_MULTIGPU_DEFAULT "OFF")
endif()

option(GRAPH_DISPATCHING_ENABLE_MULTIGPU "Enable multi-GPU support" ${GRAPH_DISPATCHING_ENABLE_MULTIGPU_DEFAULT})

if(GRAPH_DISPATCHING_ENABLE_MULTIGPU)
    if(NOT DEFINED ENV{GRAPH_DISPATCHING_GPU_COUNT})
        message(FATAL_ERROR "Multi-GPU support is enabled but the count is not given.")
    elseif($ENV{GRAPH_DISPATCHING_GPU_COUNT} LESS 2)
        message(FATAL_ERROR "Multi-GPU support is enabled but only $ENV{GRAPH_DISPATCHING_GPU_COUNT} GPUs are declared.")
    else()
        message(STATUS "Multi-GPU support is enabled (with $ENV{GRAPH_DISPATCHING_GPU_COUNT} GPUs).")
    endif()
else()
    message(STATUS "Multi-GPU support is disabled.")
endif()

# HPX 1.10.0 still tries to find Boost with a module-based approach.
# Since CMake 3.30, it's no longer available by default.
# See also https://cmake.org/cmake/help/latest/policy/CMP0167.html.
set(CMAKE_POLICY_DEFAULT_CMP0167 OLD)

# Allow the 'return' command to take arguments.
# See also https://cmake.org/cmake/help/latest/command/return.html#command:return.
set(CMAKE_POLICY_DEFAULT_CMP0140 NEW)

#---- If the compiler is Clang, we need to explicitly link to the atomic library, as
#     stated in https://clang.llvm.org/docs/Toolchain.html#atomics-library.
#     We are not sure why this is needed, i.e. why we don't get this automatically from
#     Kokkos or HPX.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(WARNING "Linking everything to 'atomic'.")
    add_link_options(-latomic)
endif()

#---- Find Kokkos.
string(JSON Kokkos_REQUIRED_VERSION  GET "${VERSION_JSON}" dependencies kokkos value)

find_package(
    Kokkos
    ${Kokkos_REQUIRED_VERSION}
    CONFIG
    REQUIRED
)

kokkos_check(DEVICES "${EXPECTED_Kokkos_DEVICES}")

target_compile_definitions(
    Kokkos::kokkoscore
    INTERFACE
        KOKKOS_IMPL_PUBLIC_INCLUDE
)

#---- Compilers like nvcc cannot compile stdexec.
if(Kokkos_CXX_COMPILER_ID STREQUAL NVIDIA AND NOT ENABLE_DOXYGEN)
    set(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS_DEFAULT OFF)
else()
    set(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS_DEFAULT ON)
endif()
option(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS "Enable stdexec tests" ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS_DEFAULT})

#---- Find Kokkos Kernels.
find_package(
    KokkosKernels
    CONFIG
    REQUIRED
)

if(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS)
    # Add 'stdexec' using the CMake Package Manager. Note that we include our 'IgnoreWarnings.cmake' right after to
    # avoid compilation issues like in https://github.com/NVIDIA/stdexec/issues/1607.
    string(JSON CPM_VERSION GET "${VERSION_JSON}" "dependencies" "cpm" "value")
    include(/opt/cpm-${CPM_VERSION}/CPM.cmake)
    string(JSON STDEXEC_VERSION GET "${VERSION_JSON}" "dependencies" "nvidia" "stdexec" "value")
    CPMAddPackage(
        NAME stdexec
        SOURCE_DIR "/opt/nvidia/stdexec-${STDEXEC_VERSION}"
        OPTIONS
            "STDEXEC_BUILD_TESTS OFF"
            "STDEXEC_BUILD_EXAMPLES OFF"
    )
else()
    # Add 'stdexec' as an include-only library, because the compiler will not be able to compiler
    # anything that the CPM would like to compile.
    # It will for instance miss the 'STDEXEC::system_context' target, see
    # https://github.com/NVIDIA/stdexec/issues/1607.
    add_library(stdexec INTERFACE)

    target_include_directories(
        stdexec
        SYSTEM
        INTERFACE
            $ENV{Stdexec_ROOT}/include
    )

    add_library(STDEXEC::stdexec ALIAS stdexec)
endif()

#---- Add Plog logging library.
include(cmake/modules/BuildPlog.cmake)

#---- Setup compiler warnings.
include(cmake/modules/IgnoreWarnings.cmake)

#---- Setup of clang-tidy if needed.
if(ENABLE_CLANG_TIDY)
    configure_file(${CMAKE_SOURCE_DIR}/.clang-tidy ${CMAKE_BINARY_DIR}/.clang-tidy @ONLY)
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy" "--config-file=${CMAKE_BINARY_DIR}/.clang-tidy")
endif()

#---- Global property that helps us keep track of files that will automatically be added
#     to our Doxygen documentation.
define_property(GLOBAL PROPERTY GraphDispatching_FILES_FOR_DOC
    BRIEF_DOCS "Files that will be added to the Doxygen documentation."
)

message(STATUS "Kokkos_CXX_COMPILER_ID is '${Kokkos_CXX_COMPILER_ID}', so GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS is set to '${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS}'.")

#---- Find GTest.
#     Note that we need GTest targets in both the 'docs' and 'tests' subdirectories,
#     so we search for it here.
string(JSON GTest_REQUIRED_VERSION GET "${VERSION_JSON}" dependencies googletest value)

find_package(
    GTest
    ${GTest_REQUIRED_VERSION}
    CONFIG
    REQUIRED
)

#---- Algorithms.
add_subdirectory(algorithms)

#---- Find kokkos-utils.
#     Note that we need kokkos-utils in both the 'tests' and 'benchmarks' subdirectories,
#     so we search for it here.
find_package(KokkosUtils CONFIG REQUIRED)

#---- Testing.
enable_testing()
add_subdirectory(kokkos_ext)
add_subdirectory(tests)

#---- Utilities.
add_subdirectory(utils)

#---- Examples.
add_subdirectory(examples)

#---- Benchmarking.
add_subdirectory(benchmarks)

#---- Documentation.
if(ENABLE_DOXYGEN)
    add_subdirectory(docs)
endif()
