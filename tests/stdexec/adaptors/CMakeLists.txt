add_one_test(NAME bulk           DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME continues_on   DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME customization  DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME ensure_started DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME let_value      DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME on             DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME split          DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME starts_on      DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME sync_wait      DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME when_all       DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
add_one_test(NAME write_env      DISABLE_KOKKOS ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})

# Add one test that checks for compile failures.
function(add_one_compile_failure_test)

    add_one_test_common(
        aocft_args
        ""
        "PASS_REGULAR_EXPRESSION"
        ""
        ${ARGN}
    )

    if(DEFINED aocft_args_EXECUTABLE_NAME)
        # Remove executable from common build targets.
        set_target_properties(
            ${aocft_args_EXECUTABLE_NAME}
            PROPERTIES
                EXCLUDE_FROM_ALL TRUE
                EXCLUDE_FROM_DEFAULT_BUILD TRUE
        )

        # Add the compile failure test.
        add_test(
            NAME ${aocft_args_EXECUTABLE_NAME}
            COMMAND ${CMAKE_COMMAND} --build . --target ${aocft_args_EXECUTABLE_NAME} --config $<CONFIG>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )

        # Set the expectation.
        set_tests_properties(
            ${aocft_args_EXECUTABLE_NAME}
            PROPERTIES
            PASS_REGULAR_EXPRESSION "${aocft_args_PASS_REGULAR_EXPRESSION}"
        )
    endif()

endfunction()

add_one_compile_failure_test(NAME write_env.fail PASS_REGULAR_EXPRESSION "The 'get_double_property_t' query must return a double." ENABLE ${GRAPH_DISPATCHING_ENABLE_STDEXEC_TESTS})
