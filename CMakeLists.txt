cmake_minimum_required(VERSION 3.23)

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

#---- Enable clang-tidy. This might bloat the compile time.
option(ENABLE_CLANG_TIDY OFF)

#---- Enable further debugging by exporting compile commands.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#---- Building Doxygen documentation is OFF by default.
option(ENABLE_DOXYGEN "Enable building Doxygen documentation" OFF)

#---- Enable multi-GPU support (tests, ...).
option(ENABLE_MULTIGPU "Enable multi-GPU support" OFF)

# HPX 1.10.0 still tries to find Boost with a module-based approach.
# Since CMake 3.30, it's no longer available by default.
# See also https://cmake.org/cmake/help/latest/policy/CMP0167.html.
set(CMAKE_POLICY_DEFAULT_CMP0167 OLD)

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

#---- Find Kokkos Kernels.
find_package(
    KokkosKernels
    CONFIG
    REQUIRED
)

#---- Setup compiler warnings.
include(cmake/modules/IgnoreWarnings.cmake)

#---- Setup of clang-tidy if needed.
if(ENABLE_CLANG_TIDY)
    configure_file(${CMAKE_SOURCE_DIR}/.clang-tidy ${CMAKE_BINARY_DIR}/.clang-tidy @ONLY)
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy" "--config-file=${CMAKE_BINARY_DIR}/.clang-tidy")
endif()

# As 'stdexec' is an include-only library, we will not use a 'FetchContent' approach because the root 'CMakeLists.txt'
# of 'stdexec' does a ton of things we just don't want at all.
add_library(stdexec INTERFACE)

target_include_directories(
    stdexec
    SYSTEM
    INTERFACE
        $ENV{Stdexec_ROOT}/include
)

add_library(stdexec::stdexec ALIAS stdexec)

#---- Global property that helps us keep track of files that will automatically be added
#     to our Doxygen documentation.
define_property(GLOBAL PROPERTY GraphDispatching_FILES_FOR_DOC
    BRIEF_DOCS "Files that will be added to the Doxygen documentation."
)

#---- Compilers like nvcc cannot compile stdexec.
if(Kokkos_CXX_COMPILER_ID STREQUAL NVIDIA AND NOT ENABLE_DOXYGEN)
    set(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS OFF)
else()
    set(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS ON)
endif()

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

#---- Benchmarking.
add_subdirectory(benchmarks)

#---- Documentation.
if(ENABLE_DOXYGEN)
    add_subdirectory(docs)
endif()
