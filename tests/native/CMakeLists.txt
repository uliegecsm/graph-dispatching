# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/APIWrappers.hpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/APIWrappers_sparse.hpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

set(enabled_backend)
set(launcher)

if(CUDA IN_LIST Kokkos_DEVICES)
    set(enabled_backend GRAPH_DISPATCHING_ENABLE_CUDA)
    set(launcher compute-sanitizer --leak-check=full)
elseif(HIP IN_LIST Kokkos_DEVICES)
    set(enabled_backend GRAPH_DISPATCHING_ENABLE_HIP)
    set(launcher)
elseif(NOT DEFINED ENABLE_DOXYGEN)
    message(FATAL_ERROR "Unsupported device backend.")
endif()

# Tests that work for both Cuda and HIP.
add_one_test(NAME APIWrappers COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher})
add_one_test(NAME diamond     COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher} ADD_BINARY_DIR)
add_one_test(NAME host_node   COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher} ADD_BINARY_DIR)
add_one_test(NAME memcpy_node COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher} ADD_BINARY_DIR)
add_one_test(NAME memory_node COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher} ADD_BINARY_DIR)
add_one_test(NAME node_enable COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher} ADD_BINARY_DIR)
add_one_test(NAME subgraph    COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher} ADD_BINARY_DIR)

if(GRAPH_DISPATCHING_ENABLE_MULTIGPU OR ENABLE_DOXYGEN)
    add_one_test(NAME multigpu COMPILE_DEFINITIONS ${enabled_backend} ADD_BINARY_DIR)
endif()

# Tests that only work for Cuda.
if(CUDA IN_LIST Kokkos_DEVICES OR ENABLE_DOXYGEN)
    # Conditional if node is not supported as of HIP 6.2.
    add_one_test(NAME conditional_if COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher} ADD_BINARY_DIR)
    # Conditional while node is not supported as of HIP 6.2.
    # Note that it triggers a weird supposedly false positive error with 'compute-sanitizer' so we don't
    # use it for the 'while' test. See https://gist.github.com/romintomasetti/6ddb4e5528fe3eabd0860ca8009e24d5.
    add_one_test(NAME conditional_while COMPILE_DEFINITIONS ${enabled_backend} ADD_BINARY_DIR)
    # This test uses cuSparse, but capture works for HIP.
    add_one_test(NAME graph_capture  COMPILE_DEFINITIONS ${enabled_backend} LAUNCHER ${launcher})
endif()

# Jobs that enable Doxygen might not have 'CUDA::cusparse' available.
if(TARGET CUDA::cusparse)
    target_link_libraries(tests_native_graph_capture PRIVATE CUDA::cusparse)
endif()
