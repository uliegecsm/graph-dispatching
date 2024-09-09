# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Kokkos_Graph_Execution.hpp)

add_subdirectory(complex_dag)
add_subdirectory(diamond)
