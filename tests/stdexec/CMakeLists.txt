# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/inline_scheduler_with_domain.hpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Utils.hpp)

add_subdirectory(adaptors)
add_subdirectory(factories)
add_subdirectory(nvcc)
add_subdirectory(queries)
add_subdirectory(schedulers)

add_one_test(NAME check_scheduler       DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME completion_signatures DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME custom_scheduler      DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME HelloWorld            DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME inline_scheduler      DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME sink_receiver         DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME value_receiver        DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
