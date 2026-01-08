# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Utils.hpp)

add_subdirectory(adaptors)
add_subdirectory(factories)
add_subdirectory(nvcc)

add_one_test(NAME check_scheduler  ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME custom_scheduler ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME HelloWorld       ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME inline_scheduler ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
