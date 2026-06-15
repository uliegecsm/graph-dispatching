#ifndef GRAPH_DISPATCHING_TESTS_SYCL_UR_CUDA_SKIP_HPP
#define GRAPH_DISPATCHING_TESTS_SYCL_UR_CUDA_SKIP_HPP

#if defined(KOKKOS_ENABLE_SYCL)
#include "sycl/version.hpp"
#endif

/**
 * Many tests will fail under the SYCL Unified Runtime (UR) with CUDA enabled.
 *
 * Use this macro to skip them.
 *
 * See https://github.com/intel/llvm/issues/22322.
 */
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
    #if !defined(__SYCL_COMPILER_VERSION)
        #error "SYCL compiler version is not defined."
    #elif __SYCL_COMPILER_VERSION == 20260319
        #define GRAPH_DISPATCHING_TESTS_SYCL_UR_CUDA_SKIP GTEST_SKIP() << "Known to fail on SYCL with Unified Runtime CUDA adaptor.";
    #else
        #define GRAPH_DISPATCHING_TESTS_SYCL_UR_CUDA_SKIP
    #endif
#else
    #define GRAPH_DISPATCHING_TESTS_SYCL_UR_CUDA_SKIP
#endif

#endif // GRAPH_DISPATCHING_TESTS_SYCL_UR_CUDA_SKIP_HPP
