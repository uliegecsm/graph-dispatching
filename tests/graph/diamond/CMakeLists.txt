# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

add_one_test(NAME kokkos)
# add_one_test(NAME outlook)
add_one_test(NAME stdexec ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
