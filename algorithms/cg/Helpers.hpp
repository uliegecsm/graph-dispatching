#ifndef GRAPH_DISPATCHING_ALGORITHMS_CG_HELPERS_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_CG_HELPERS_HPP

namespace algorithms::cg
{

/// As stated in https://docs.nvidia.com/cuda/cublas/#scalar-parameters, @c cuBLAS can deal with scalar parameters in 2 ways:
///     * @c CUBLAS_POINTER_MODE_HOST
///     * @c CUBLAS_POINTER_MODE_DEVICE
/// This function checks that we are in the @c CUBLAS_POINTER_MODE_HOST mode, which implies that:
///     1. For methods that take scalar parameters (*e.g.* @c axpy), they don't need to be placed in managed memory.
///        The kernel will use its own copy of the variables.
///     2. For methods that return a scalar value (*e.g. @c dot), the @c cuBLAS call will block the @c CPU thread until the kernel is finished.
///
/// So using scalar values on the stack is fine, and for an implementation of the @c CG that uses a single execution space instance, it's even
/// better than using device or host pinned views for storing intermediate variables (because it makes the whole thing more readable).
///
/// Note that the @c CUBLAS_POINTER_MODE_DEVICE mode can be interesting. Quoting:
///     For example, this situation can arise when iterative methods for solution of linear systems and eigenvalue problems are implemented using the cuBLAS library.
void check_cublas_uses_host_pointer_mode()
{
#if defined(KOKKOS_ENABLE_CUDA)
    #if !defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
        #error "Kokkos Kernels TPL cuBLAS not enabled."
    #endif
    const KokkosBlas::Impl::CudaBlasSingleton& singleton = KokkosBlas::Impl::CudaBlasSingleton::singleton();
    cublasPointerMode_t mode;
    KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasGetPointerMode(singleton.handle, &mode));
    if(mode != CUBLAS_POINTER_MODE_HOST)
        Kokkos::abort("cuBLAS pointer mode is not host.");
#else
    static_assert("Not implemented.");
#endif
}

/**
 * @brief Use this helper to better describe which region is being closed.
 *
 * It would also prevent you from deleting the push but keeping the pop.
 *
 * @note It does not store the name of the region to ensure that the footprint
 *       of this helper is minimum.
 *
 * @note In debug mode, we could add a check in the destructor that the @ref pop has been
 *       called.
 */
struct Region
{
    template <typename T>
    explicit Region(T&& arg) { Kokkos::Profiling::pushRegion(std::forward<T>(arg)); }

    static void pop() { Kokkos::Profiling::popRegion(); }
};

} // namespace algorithms::cg

#endif // GRAPH_DISPATCHING_ALGORITHMS_CG_HELPERS_HPP
