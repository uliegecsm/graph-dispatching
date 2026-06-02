#ifndef GRAPH_DISPATCHING_ALGORITHMS_PCG_SINGLEQUEUE_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_PCG_SINGLEQUEUE_HPP

#include "plog/Log.h"

#include "Kokkos_Profiling_ScopedRegion.hpp"

#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_spmv.hpp"

#include "algorithms/cg/Base.hpp"
#include "algorithms/cg/Helpers.hpp"
#include "utils/pool.hpp"

namespace algorithms::pcg
{

/**
 * @brief Preconditioned conjugate gradient solver with a single @c Kokkos execution space instance.
 *
 * See @cite SAAD2003 (algorithm 9.1 in section 9.2 on page 263).
 *
 * @note @c SpmvType, @c DotType and @c AxpbyType are used only during the @c while loop.
 */
template <
    typename MatrixType, typename VectorType, typename PreconditionerType,
    typename SpmvType,
    typename DotType,
    typename AxpbyType
>
struct PCGSingleQueue : public algorithms::cg::CGBase<MatrixType, VectorType>
{
public:
    using base_t = algorithms::cg::CGBase<MatrixType, VectorType>;

    using typename base_t::dot_t;
    using typename base_t::mag_t;

public:
    MatrixType mat;
    VectorType rhs;

protected:
    PreconditionerType preconditioner;
    VectorType res; //!< Residual.
    VectorType res_p; //!< Residual of the preconditioned system.
    VectorType dir; //!< Direction of search.
    VectorType mat_dir; //!< Placeholder for the product of @ref mat with @ref dir.

public:
    template <Kokkos::ExecutionSpace Exec, typename MatType, typename RhsType>
    PCGSingleQueue(const Exec& exec, MatType&& mat_, RhsType&& rhs_)
        : mat(std::forward<MatType>(mat_)),
          rhs(std::forward<RhsType>(rhs_)),
          preconditioner{exec, mat},
          res    (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual"),                          rhs.size()),
          res_p  (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual of preconditioned system"), rhs.size()),
          dir    (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "dir"),                               rhs.size()),
          mat_dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "mat * dir"),                         rhs.size())
    {}

    //! Allow the user to interact with the preconditioner, *e.g.* to update its parameters between 2 subsequent calls to @ref apply.
    auto& get_preconditioner() & { return preconditioner; }

    /// We explicitly don't partition @p exec.
    /// This decision is aligned with how @c KokkosKernels would use the incoming @p exec.
    template <Kokkos::ExecutionSpace Exec>
    std::tuple<mag_t, decltype(base_t::Parameters::max_iters)> apply(const utils::ExecutionSpacePool<Exec>& pool, const VectorType& sol, const base_t::Parameters& params) const
    {
        PLOG_INFO << "PCGSingleQueue(apply): starting...";

        typename base_t::template spmv_handle_t<Exec> handle {};

#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::same_as<Exec, Kokkos::Cuda>) {
            algorithms::cg::check_cublas_uses_host_pointer_mode();
        }
#endif

        if(pool.size() < 2) Kokkos::abort("The pool size must be at least 2.");
        const auto exec_A = pool.get(0);
        const auto exec_B = pool.get(1);

        //! Pre-compute the residual.
        Kokkos::deep_copy(exec_A, res, rhs);
        KokkosSparse::spmv(exec_A, &handle, "N", -1., mat, sol, 1., res);

        //! Placeholder for the residual L2 norm.
        mag_t res_nrm2 = Kokkos::sqrt(Kokkos::abs(KokkosBlas::dot(exec_A, res, res)));
        if(res_nrm2 < params.tolerance) return {res_nrm2, 0};

        //! Pre-compute the residual of the preconditioned system.
        preconditioner.apply(exec_A, res_p, res);

        //! Compute the dot product of both residuals.
        dot_t res_p_res_dot_old = KokkosBlas::dot(exec_A, res_p, res);

        //! Direction of search is set to the residual of the preconditioned system.
        Kokkos::deep_copy(exec_A, dir, res_p);

        /// For @c Cuda, the @c dot will not end up in a @c cuBLAS call (see https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/tpls/KokkosBlas1_dot_tpl_spec_avail.hpp#L81-L90).
        /// Therefore, @c KokkosBlas::dot will end up doing a @c Kokkos parallel reduce.
        /// If the result variable is a host scalar, it ends up making 2 fences (instead of one, see below):
        ///     - in @c Kokkos parallel reduce itself (see *e.g.* https://github.com/kokkos/kokkos/blob/c2a5c01699048e80a8ddce9d99c0050b70238b7c/core/src/Cuda/Kokkos_Cuda_Parallel_MDRange.hpp#L455-L470)
        ///     - in @c KokkosBlas (see *e.g.* https://github.com/kokkos/kokkos-kernels/blob/9bca19c85b88aeca97209ec7cde858447e16696c/blas/src/KokkosBlas1_dot.hpp#L107)
        /// To ensure we control fencing, we'll use a host pinned variable for intermediate @c dot results.
        const Kokkos::View<dot_t, Kokkos::SharedHostPinnedSpace> pinned("intermediate dot result");

        const Kokkos::Profiling::ScopedRegion loop("PCGSingleQueue - loop");

        //! Loop until the norm of the residual is smaller than @p tol or the maximum number of iterations is reached.
        decltype(base_t::Parameters::max_iters) iter = 0;
        while(res_nrm2 > params.tolerance && iter < params.max_iters)
        {
#if defined(GRAPH_DISPATCHING_ALGORITHMS_PCG_PCGSINGLEQUEUE_ENABLE_SCOPEDREGION_IN_LOOP)
            const Kokkos::Profiling::ScopedRegion region("PCGSingleQueue - iter " + std::to_string(iter));
#endif

            //! Compute @c alpha.
            SpmvType{}(exec_A, &handle, "N", 1., mat, dir, 0., mat_dir);

            DotType{}(exec_A, pinned, dir, mat_dir);

            exec_A.fence("waiting for dot (quadratic)");

            const auto alpha = res_p_res_dot_old / pinned();

            //! Update the solution candidate and residual.
            AxpbyType{}(exec_A,   alpha, dir,     1., sol);
            AxpbyType{}(exec_B, - alpha, mat_dir, 1., res);

            //! At this point, we can already check the condition and exit.
            DotType{}(exec_B, pinned, res, res);

            exec_B.fence("waiting for dot (norm)");

            res_nrm2 = Kokkos::sqrt(Kokkos::abs(pinned()));

            if(res_nrm2 > params.tolerance)
            {
                //! Update the residual of the preconditioned system.
                preconditioner.apply(exec_B, res_p, res);

                //! Compute the dot product of the residuals.
                DotType{}(exec_B, pinned, res_p, res);

                exec_B.fence("waiting for dot (residuals)");

                //! Compute @c beta.
                const auto beta = pinned() / res_p_res_dot_old;

                //! Update search direction.
                AxpbyType{}(exec_A, 1., res_p, beta, dir);

                res_p_res_dot_old = pinned();
            }

            PLOG_INFO << "PCGSingleQueue(apply): iteration " << iter << ", res nrm2 " << res_nrm2;

            ++iter;
        }

        return {res_nrm2, iter};
    }
};

} // namespace algorithms::pcg

#endif // GRAPH_DISPATCHING_ALGORITHMS_PCG_SINGLEQUEUE_HPP
