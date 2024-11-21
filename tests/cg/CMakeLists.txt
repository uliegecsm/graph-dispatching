# Don't forget export KokkosKernels_ROOT=$PWD/kokkos-kernels-compiled/ROCm-HIP
find_package(
    KokkosKernels
    CONFIG
    REQUIRED
)

add_one_test(NAME outlook ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels)
