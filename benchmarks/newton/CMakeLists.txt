add_one_benchmark(NAME compare ADDITIONAL_LINK_LIBRARIES KokkosKernels::kokkoskernels Kokkos::kokkosutils plog::plog)
sycl_ur_cuda_skip(benchmarks_newton_compare)
