# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/APIWrappers.hpp)

add_one_test(NAME APIWrappers COMPILE_DEFINITIONS GRAPH_DISPATCHING_ENABLE_CUDA LAUNCHER compute-sanitizer)
