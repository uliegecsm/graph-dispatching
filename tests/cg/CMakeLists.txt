# Don't forget export KokkosKernels_ROOT=$PWD/kokkos-kernels-compiled/ROCm-HIP
find_package(
    KokkosKernels
    CONFIG
    REQUIRED
)

# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

add_one_test(NAME regular        ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels)
add_one_test(NAME outlook        ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels)
add_one_test(NAME outlook.memcpy ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels)
