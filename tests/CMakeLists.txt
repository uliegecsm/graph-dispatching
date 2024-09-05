#---- Helpers.
include(${CMAKE_SOURCE_DIR}/cmake/functions/path.cmake)

#---- Find GTest.
string(JSON GTest_REQUIRED_VERSION GET "${VERSION_JSON}" dependencies googletest value)

find_package(
    GTest
    ${GTest_REQUIRED_VERSION}
    CONFIG
    REQUIRED
)

#---- Function to add a test.
function(add_one_test)

    cmake_parse_arguments(aot_args "" "NAME" "ADDITIONAL_LINK_LIBRARIES" ${ARGN})

    # Set executable name and primary source file.
    cmake_path(SET SOURCE_FILE "test_${aot_args_NAME}.cpp")
    cmake_path(ABSOLUTE_PATH SOURCE_FILE)

    get_parent_relpath_repr(DIRECTORY "${PROJECT_SOURCE_DIR}" FILENAME "${SOURCE_FILE}")

    set(EXECUTABLE_NAME "${gprr_PARENT_RELPATH_REPR}_${aot_args_NAME}")

    # Add source file to the documentation list.
    set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${SOURCE_FILE})

    # Create test executable.
    add_executable(${EXECUTABLE_NAME})

    # Add the source file to the executable.
    target_sources(
        ${EXECUTABLE_NAME}
        PRIVATE
            ${SOURCE_FILE}
    )

    # Compile options.
    target_compile_options(
        ${EXECUTABLE_NAME}
        PRIVATE
            -Wall
            -Wextra
            -Werror
    )

    # Link the executable to GTest.
    target_link_libraries(
        ${EXECUTABLE_NAME}
        PRIVATE
            test_common
            stdexec::stdexec
    )

    if(DEFINED aot_args_ADDITIONAL_LINK_LIBRARIES)
        target_link_libraries(
            ${EXECUTABLE_NAME}
            PRIVATE
               ${aot_args_ADDITIONAL_LINK_LIBRARIES}
        )
    endif()

    # Allow tests to reference files in this repository directly.
    target_include_directories(
        ${EXECUTABLE_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}
    )

    # Add the test to CTest.
    add_test(
        NAME    ${EXECUTABLE_NAME}
        COMMAND $<TARGET_FILE:${EXECUTABLE_NAME}>
    )

endfunction()

#---- Common test object library.
add_library(test_common OBJECT)
target_sources(test_common PRIVATE main.cpp)
target_link_libraries(test_common PUBLIC GTest::gtest Kokkos::kokkoscore)

# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Utils.hpp)

# Include subdirectories.
add_subdirectory(graph)
add_subdirectory(hpx)
add_subdirectory(stdexec)
