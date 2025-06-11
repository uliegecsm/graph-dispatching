set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

add_one_test(NAME single_queue        ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels Kokkos::kokkosutils)
add_one_test(NAME single_queue_no_tpl ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels Kokkos::kokkosutils)
