if(CUDA IN_LIST Kokkos_DEVICES OR ENABLE_DOXYGEN)
    add_one_test(NAME while)
endif()
