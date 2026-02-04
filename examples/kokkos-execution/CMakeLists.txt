function(add_one_example NAME)

    cmake_parse_arguments(aoe_args "" "ENABLE" "" ${ARGN})

    # Set executable name and primary source file.
    cmake_path(SET SOURCE_FILE "example_${NAME}.cpp")
    cmake_path(ABSOLUTE_PATH SOURCE_FILE)

    get_parent_relpath_repr(DIRECTORY "${PROJECT_SOURCE_DIR}" FILENAME "${SOURCE_FILE}")

    set(EXECUTABLE_NAME "${gprr_PARENT_RELPATH_REPR}_${NAME}")

    # Possibly disable this example. This also means the executable is not compiled.
    if(DEFINED aoe_args_ENABLE AND NOT aoe_args_ENABLE)
        message(WARNING "Skipping example '${EXECUTABLE_NAME}'.")
        return()
    else()
        message(STATUS "Adding example '${EXECUTABLE_NAME}'.")
    endif()

    # Add source file to the documentation list.
    set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${SOURCE_FILE})

    # Create executable.
    add_executable(${EXECUTABLE_NAME})

    # Add the source file to the executable.
    target_sources(${EXECUTABLE_NAME} PRIVATE ${SOURCE_FILE})

    # Link the executable to 'test_common'.
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE test_common)

    # Always link stdexec, kokkos-utils and plog.
    target_link_libraries(
        ${EXECUTABLE_NAME}
        PRIVATE
            STDEXEC::stdexec
            Kokkos::kokkosutils
            plog
    )

    # Allow tests to reference files in this repository directly.
    target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CMAKE_SOURCE_DIR})

    # Add it to the test set.
    add_test(NAME ${EXECUTABLE_NAME} COMMAND $<TARGET_FILE:${EXECUTABLE_NAME}> --gtest_fail_if_no_test_linked)
endfunction()

add_one_example(then ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
