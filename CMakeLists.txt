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

#---- Find Kokkos.
string(JSON Kokkos_REQUIRED_VERSION  GET "${VERSION_JSON}" dependencies kokkos value)

find_package(
    Kokkos
    ${Kokkos_REQUIRED_VERSION}
    CONFIG
    REQUIRED
)

kokkos_check(DEVICES "${EXPECTED_Kokkos_DEVICES}")

# As 'stdexec' is an include-only library, we will not use a 'FetchContent' approach because the root 'CMakeLists.txt'
# of 'stdexec' does a ton of things we just don't want at all.
add_library(stdexec INTERFACE)

target_include_directories(
    stdexec
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
if(CUDA IN_LIST Kokkos_DEVICES)
    set(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS OFF)
else()
    set(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS ON)
endif()

#---- Testing.
enable_testing()
add_subdirectory(tests)

#---- Documentation.
add_subdirectory(docs)
