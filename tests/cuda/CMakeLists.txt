# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/APIWrappers.hpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

add_one_test(NAME APIWrappers    COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer)
add_one_test(NAME conditional_if COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer ADD_BINARY_DIR)
add_one_test(NAME diamond        COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer ADD_BINARY_DIR)
add_one_test(NAME graph_capture  COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer ADD_BINARY_DIR)
add_one_test(NAME host_node      COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer ADD_BINARY_DIR)
add_one_test(NAME memory_node    COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer ADD_BINARY_DIR)
add_one_test(NAME node_enable    COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer ADD_BINARY_DIR)
add_one_test(NAME subgraph       COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer ADD_BINARY_DIR)

# Jobs that enable Doxygen might not have 'CUDA::cusparse' available.
if(TARGET CUDA::cusparse)
    target_link_libraries(tests_cuda_graph_capture PRIVATE CUDA::cusparse)
endif()
