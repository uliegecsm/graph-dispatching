#---- Helpers.
include(${CMAKE_SOURCE_DIR}/cmake/functions/path.cmake)

#---- Function to add a test.
function(add_one_test)

    cmake_parse_arguments(
        aot_args
        "DISABLE_KOKKOS;ADD_BINARY_DIR"
        "NAME;ENABLE"
        "ADDITIONAL_LINK_LIBRARIES;COMPILE_DEFINITIONS;LAUNCHER"
        ${ARGN}
    )

    # Possibly disable this test. This also means the executable is not compiled.
    if(DEFINED aot_args_ENABLE AND NOT aot_args_ENABLE)
        message(WARNING "Skipping test ${aot_args_NAME}.")
        return()
    endif()

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

    # Compile definitions.
    if(DEFINED aot_args_COMPILE_DEFINITIONS)
        target_compile_definitions(${EXECUTABLE_NAME} PRIVATE ${aot_args_COMPILE_DEFINITIONS})
    endif()

    if(DEFINED aot_args_ADD_BINARY_DIR)
        target_compile_definitions(${EXECUTABLE_NAME} PRIVATE CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    # Link the executable to Google Test main
    # if Kokkos is disabled, to 'test_common' otherwise.
    # Always link to 'stdexec'.
    if(aot_args_DISABLE_KOKKOS)
        message(STATUS "Kokkos is disabled for '${EXECUTABLE_NAME}'.")
        target_link_libraries(${EXECUTABLE_NAME} PRIVATE GTest::gtest_main)
    else()
        message(STATUS "Kokkos is enabled for '${EXECUTABLE_NAME}'.")
        target_link_libraries(${EXECUTABLE_NAME} PRIVATE test_common)
    endif()
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE stdexec::stdexec)

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
    if(DEFINED aot_args_LAUNCHER)
        set(TEST_COMMAND ${aot_args_LAUNCHER})
    else()
        set(TEST_COMMAND)
    endif()
    list(APPEND TEST_COMMAND $<TARGET_FILE:${EXECUTABLE_NAME}>)
    add_test(NAME ${EXECUTABLE_NAME} COMMAND ${TEST_COMMAND})

    # Ensure that GTest will shuffle the tests.
    set_property(TEST ${EXECUTABLE_NAME} APPEND PROPERTY ENVIRONMENT "GTEST_SHUFFLE=1")
    set_property(TEST ${EXECUTABLE_NAME} APPEND PROPERTY ENVIRONMENT "GTEST_RANDOM_SEED=0")
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

# Include subdirectories.
add_subdirectory(cg)
add_subdirectory(compiler)
add_subdirectory(exec)
add_subdirectory(graph)
add_subdirectory(hpx)
add_subdirectory(kokkos_ext)
if(CUDA IN_LIST Kokkos_DEVICES OR HIP IN_LIST Kokkos_DEVICES OR ENABLE_DOXYGEN)
    add_subdirectory(native)
endif()
if(CUDA IN_LIST Kokkos_DEVICES OR ENABLE_DOXYGEN)
    add_subdirectory(nvexec)
endif()
add_subdirectory(stdexec)
