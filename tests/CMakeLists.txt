#---- Helpers.
include(${CMAKE_SOURCE_DIR}/cmake/functions/path.cmake)

#---- Function to add a test (common steps).
function(add_one_test_common PREFIX OPTIONS ONE_VALUE_KEYWORDS MULTI_VALUE_KEYWORDS)

    cmake_parse_arguments(
        PARSE_ARGV 1
        ${PREFIX}
        "DISABLE_KOKKOS;ADD_BINARY_DIR;${OPTIONS}"
        "NAME;ENABLE;${ONE_VALUE_KEYWORDS}"
        "ADDITIONAL_LINK_LIBRARIES;COMPILE_DEFINITIONS;${MULTI_VALUE_KEYWORDS}"
    )

    # Possibly disable this test. This also means the executable is not compiled.
    if(DEFINED ${PREFIX}_ENABLE AND NOT ${PREFIX}_ENABLE)
        message(WARNING "Skipping test '${${PREFIX}_NAME}'.")
        return()
    else()
        message(STATUS "Adding test '${${PREFIX}_NAME}'.")
    endif()

    # Set executable name and primary source file.
    cmake_path(SET SOURCE_FILE "test_${${PREFIX}_NAME}.cpp")
    cmake_path(ABSOLUTE_PATH SOURCE_FILE)

    get_parent_relpath_repr(DIRECTORY "${PROJECT_SOURCE_DIR}" FILENAME "${SOURCE_FILE}")

    set(${PREFIX}_EXECUTABLE_NAME "${gprr_PARENT_RELPATH_REPR}_${${PREFIX}_NAME}")

    # Add source file to the documentation list.
    set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${SOURCE_FILE})

    # Create test executable.
    add_executable(${${PREFIX}_EXECUTABLE_NAME})

    # Add the source file to the executable.
    target_sources(
        ${${PREFIX}_EXECUTABLE_NAME}
        PRIVATE
            ${SOURCE_FILE}
    )

    # Compile definitions.
    if(DEFINED ${PREFIX}_COMPILE_DEFINITIONS)
        message(STATUS "Adding compile definitions '${${PREFIX}_COMPILE_DEFINITIONS}' for '${${PREFIX}_EXECUTABLE_NAME}'.")
        target_compile_definitions(${${PREFIX}_EXECUTABLE_NAME} PRIVATE ${${PREFIX}_COMPILE_DEFINITIONS})
    endif()

    if(DEFINED ${PREFIX}_ADD_BINARY_DIR AND ${PREFIX}_ADD_BINARY_DIR)
        message(STATUS "Adding 'CMAKE_CURRENT_BINARY_DIR' compile definition for '${${PREFIX}_EXECUTABLE_NAME}'.")
        target_compile_definitions(${${PREFIX}_EXECUTABLE_NAME} PRIVATE CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    # Link the executable to Google Test main
    # if Kokkos is disabled, to 'test_common' otherwise.
    # Always link to 'stdexec'.
    if(${PREFIX}_DISABLE_KOKKOS)
        message(STATUS "Kokkos is disabled for '${${PREFIX}_EXECUTABLE_NAME}'.")
        target_link_libraries(${${PREFIX}_EXECUTABLE_NAME} PRIVATE GTest::gtest_main)
    else()
        message(STATUS "Kokkos is enabled for '${${PREFIX}_EXECUTABLE_NAME}'.")
        target_link_libraries(${${PREFIX}_EXECUTABLE_NAME} PRIVATE test_common)
    endif()

    if(GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS)
        target_link_libraries(${${PREFIX}_EXECUTABLE_NAME} PRIVATE STDEXEC::stdexec)
    endif()

    if(DEFINED ${PREFIX}_ADDITIONAL_LINK_LIBRARIES)
        target_link_libraries(
            ${${PREFIX}_EXECUTABLE_NAME}
            PRIVATE
               ${${PREFIX}_ADDITIONAL_LINK_LIBRARIES}
        )
    endif()

    # Always allow logging.
    target_link_libraries(${${PREFIX}_EXECUTABLE_NAME} PRIVATE plog)

    # Allow tests to reference files in this repository directly.
    target_include_directories(
        ${${PREFIX}_EXECUTABLE_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}
    )

    # Allow calling scopes to access variables.
    set(PROPAGATE_LIST EXECUTABLE_NAME)
    list(APPEND PROPAGATE_LIST ${OPTIONS})
    list(APPEND PROPAGATE_LIST ${ONE_VALUE_KEYWORDS})
    list(APPEND PROPAGATE_LIST ${MULTI_VALUE_KEYWORDS})
    list(TRANSFORM PROPAGATE_LIST PREPEND "${PREFIX}_")
    return(PROPAGATE ${PROPAGATE_LIST})

endfunction()

#---- Function to add a test ("regular" way, i.e. launch the executable).
function(add_one_test)

    add_one_test_common(
        aot_args
        ""
        ""
        "LAUNCHER"
        ${ARGN}
    )

    if(DEFINED aot_args_EXECUTABLE_NAME)
        # Add the test to CTest.
        if(DEFINED aot_args_LAUNCHER)
            message(STATUS "Using '${aot_args_LAUNCHER}' as launcher for test '${aot_args_EXECUTABLE_NAME}'.")
            set(TEST_COMMAND ${aot_args_LAUNCHER})
        else()
            set(TEST_COMMAND)
        endif()
        list(APPEND TEST_COMMAND $<TARGET_FILE:${aot_args_EXECUTABLE_NAME}>)
        add_test(NAME ${aot_args_EXECUTABLE_NAME} COMMAND ${TEST_COMMAND})

        # Ensure that GTest will shuffle the tests.
        set_property(TEST ${aot_args_EXECUTABLE_NAME} APPEND PROPERTY ENVIRONMENT "GTEST_SHUFFLE=1")
        set_property(TEST ${aot_args_EXECUTABLE_NAME} APPEND PROPERTY ENVIRONMENT "GTEST_RANDOM_SEED=0")
    endif()

endfunction()

#---- Common test object library.
add_library(test_common OBJECT)
target_sources(test_common PRIVATE main.cpp)
target_link_libraries(test_common PUBLIC GTest::gtest Kokkos::kokkoscore)

# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/CallbackMatchers.hpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Functors.hpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Utils.hpp)
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/utils/LoadCheckAdd.hpp)

# Include subdirectories.
add_subdirectory(cg)
add_subdirectory(compiler)
add_subdirectory(exec)
add_subdirectory(graph)
add_subdirectory(hpx)
add_subdirectory(kokkos)
add_subdirectory(kokkos_ext)
if(CUDA IN_LIST Kokkos_DEVICES OR HIP IN_LIST Kokkos_DEVICES OR ENABLE_DOXYGEN)
    add_subdirectory(native)
endif()
if(CUDA IN_LIST Kokkos_DEVICES OR ENABLE_DOXYGEN)
    add_subdirectory(nvexec)
endif()
add_subdirectory(pcg)
add_subdirectory(plog)
add_subdirectory(stdexec)
