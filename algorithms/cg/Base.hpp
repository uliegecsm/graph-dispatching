#ifndef GRAPH_DISPATCHING_ALGORITHMS_CG_BASE_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_CG_BASE_HPP

namespace algorithms::cg
{

/**
 * @brief Conjugate gradient solver base.
 *
 * References:
 *  - https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
 */
template <typename MatrixType, typename VectorType>
struct CGBase
{
    //! Result of @c nrm2.
    using mag_t = typename Kokkos::Details::InnerProductSpaceTraits<typename VectorType::non_const_value_type>::mag_type;

    //! Result of @c dot.
    using dot_t = typename Kokkos::Details::InnerProductSpaceTraits<typename VectorType::non_const_value_type>::dot_type;

    //! @c KokkosSparse::spmv handle.
    template <typename Exec>
    using spmv_handle_t = KokkosSparse::SPMVHandle<Exec, MatrixType, VectorType, VectorType>;
};

} // namespace algorithms::cg

#endif // GRAPH_DISPATCHING_ALGORITHMS_CG_BASE_HPP
