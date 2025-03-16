add_library(kokkosext INTERFACE)
target_link_libraries(kokkosext INTERFACE plog::plog)

add_subdirectory(execution_space)
