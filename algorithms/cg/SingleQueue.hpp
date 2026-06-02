#ifndef GRAPH_DISPATCHING_ALGORITHMS_CG_SINGLEQUEUE_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_CG_SINGLEQUEUE_HPP

#include "Kokkos_Profiling_ScopedRegion.hpp"

#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_spmv.hpp"

#include "algorithms/cg/Base.hpp"
#include "algorithms/cg/Helpers.hpp"
#include "utils/pool.hpp"

namespace algorithms::cg
{

/**
 * @brief Conjugate gradient solver with a single @c Kokkos execution space instance.
 *
 * @note @c SpmvType, @c DotType and @c AxpbyType are used only during the @c while loop.
 */
template <
    typename MatrixType, typename VectorType,
    typename SpmvType,
    typename DotType,
    typename AxpbyType
>
struct CGSingleQueue : public CGBase<MatrixType, VectorType>
{
    using base_t = CGBase<MatrixType, VectorType>;

    using typename base_t::dot_t;
    using typename base_t::mag_t;

    MatrixType mat;
    VectorType rhs;

    /// We explicitly don't partition @p exec.
    /// This decision is aligned with how @c KokkosKernels would use the incoming @p exec.
    template <Kokkos::ExecutionSpace Exec, typename SizeType = typename Exec::size_type>
    std::tuple<mag_t, SizeType> apply(const utils::ExecutionSpacePool<Exec>& pool, const VectorType& sol, const typename base_t::Parameters& params) const
    {
        typename base_t::template spmv_handle_t<Exec> handle {};

#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::same_as<Exec, Kokkos::Cuda>) {
            check_cublas_uses_host_pointer_mode();
        }
#endif

        if(pool.size() < 2) Kokkos::abort("The pool size must be at least 2.");
        const auto& exec_A = pool.get(0);
        const auto& exec_B = pool.get(1);

        //! Pre-compute the residual.
        VectorType res(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec_A, "residual"), rhs.size());
        Kokkos::deep_copy(exec_A, res, rhs);
        KokkosSparse::spmv(exec_A, &handle, "N", -1., mat, sol, 1., res);

        //! Compute the residual norm, because it might already be all good.
        dot_t res_dot_old = KokkosBlas::dot(exec_A, res, res);

        //! Placeholder for the residual L2 norm.
        mag_t res_nrm2 = Kokkos::sqrt(Kokkos::abs(res_dot_old));
        if(res_nrm2 < params.tolerance) return {res_nrm2, 0};

        //! Direction of search is set to the residual.
        const VectorType dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec_A, "dir"), rhs.size());
        Kokkos::deep_copy(exec_A, dir, res);

        //! Placeholder for the product of @ref mat with the direction of search.
        const VectorType mat_dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec_A, "mat * dir"), rhs.size());

        /// For @c Cuda, the @c dot will not end up in a @c cuBLAS call (see https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/tpls/KokkosBlas1_dot_tpl_spec_avail.hpp#L81-L90).
        /// Therefore, @c KokkosBlas::dot will end up doing a @c Kokkos parallel reduce.
        /// If the result variable is a host scalar, it ends up making 2 fences (instead of one, see below):
        ///     - in @c Kokkos parallel reduce itself (see *e.g.* https://github.com/kokkos/kokkos/blob/c2a5c01699048e80a8ddce9d99c0050b70238b7c/core/src/Cuda/Kokkos_Cuda_Parallel_MDRange.hpp#L455-L470)
        ///     - in @c KokkosBlas (see *e.g.* https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/src/KokkosBlas1_dot.hpp#L107)
        /// To ensure we control fencing, we'll use a host pinned variable for intermediate @c dot results.
        const Kokkos::View<dot_t, Kokkos::SharedHostPinnedSpace> pinned("intermediate dot result");

        const Kokkos::Profiling::ScopedRegion loop("CGSingleQueue - loop");

        //! Loop until the norm of the residual is smaller than @p tol or the maximum number of iterations is reached.
        SizeType iter = 0;
        while(res_nrm2 > params.tolerance && iter < params.max_iters)
        {
#if defined(GRAPH_DISPATCHING_ALGORITHMS_CG_CGSINGLEQUEUE_ENABLE_SCOPEDREGION_IN_LOOP)
            const Kokkos::Profiling::ScopedRegion region("CGSingleQueue - iter " + std::to_string(iter));
#endif

            //! Compute @c alpha.
            SpmvType{}(exec_A, &handle, "N", 1., mat, dir, 0., mat_dir);

            DotType{}(exec_A, pinned, dir, mat_dir);

            exec_A.fence("waiting for dot (quadratic)");

            const auto alpha = res_dot_old / pinned();

            /// Update the solution candidate and residual.
            /// These two updates are independent, but since we have only one execution space instance, all we gain is
            /// the launch overhead.
            AxpbyType{}(exec_A,   alpha, dir,     1., sol);
            AxpbyType{}(exec_B, - alpha, mat_dir, 1., res);

            //! At this point, we can already check the condition and exit.
            DotType{}(exec_B, pinned, res, res);

            exec_B.fence("waiting for dot (norm)");

            if((res_nrm2 = Kokkos::sqrt(Kokkos::abs(pinned()))) > params.tolerance)
            {
                /// Compute @c beta.
                /// @note Though it should be purely real, floating point errors add up and it has a very tiny imaginary part.
                const auto beta = pinned() / res_dot_old;

                //! Update search direction.
                AxpbyType{}(exec_A, 1., res, beta, dir);

                res_dot_old = pinned();
            }

            ++iter;
        }

        return {res_nrm2, iter};
    }
};

} // namespace algorithms::cg

#endif // GRAPH_DISPATCHING_ALGORITHMS_CG_SINGLEQUEUE_HPP
