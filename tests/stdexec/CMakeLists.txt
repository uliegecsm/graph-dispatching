add_subdirectory(adaptors)

add_one_test(NAME custom_scheduler ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME HelloWorld       ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
