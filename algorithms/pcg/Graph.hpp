#ifndef GRAPH_DISPATCHING_ALGORITHMS_PCG_GRAPH_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_PCG_GRAPH_HPP

#include "plog/Log.h"

#include "Kokkos_Graph.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_spmv.hpp"

#include "algorithms/cg/Base.hpp"
#include "algorithms/cg/Helpers.hpp"

namespace algorithms::pcg
{

//! Compilers don't like @c KOKKOS_LAMBDA that captures members.
template <typename AlphaType, typename AType, typename BType>
struct Alpha
{
    AlphaType alpha, alpha_neg;
    AType a;
    BType b;

    KOKKOS_FUNCTION
    void operator()() const
    {
        alpha()     = a() / b();
        alpha_neg() = -alpha();
    }
};

/**
 * @brief Preconditioned conjugate gradient solver with a @c Kokkos::Graph.
 *
 * See @cite SAAD2003 (algorithm 9.1 in section 9.2 on page 263).
 */
template <typename MatrixType, typename VectorType, typename GraphType, typename PreconditionerType>
struct PCGGraph : public algorithms::cg::CGBase<MatrixType, VectorType>
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

    //! @name Temporaries.
    ///@{
    Kokkos::View<dot_t,    typename GraphType::execution_space> res_res_p_dot_old;
    Kokkos::View<dot_t[3], typename GraphType::execution_space> scalars_on_device;
    Kokkos::View<dot_t,    Kokkos::SharedHostPinnedSpace      > res_dot;
    ///@}

    mutable std::optional<GraphType> graph = std::nullopt; //!< For now it is mutable because the first time we enter @ref apply we build and instantiate it.

public:
    template <Kokkos::ExecutionSpace Exec, typename MatType, typename RhsType>
    PCGGraph(const Exec& exec, MatType&& mat_, RhsType&& rhs_)
        : mat(std::forward<MatType>(mat_)),
          rhs(std::forward<RhsType>(rhs_)),
          preconditioner(exec, mat),
          res    (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual"),                          rhs.size()),
          res_p  (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual of preconditioned system"), rhs.size()),
          dir    (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "dir"),                               rhs.size()),
          mat_dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "mat * dir"),                         rhs.size()),
          res_res_p_dot_old(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "res_res_p_dot_old is needed on device")),
          scalars_on_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "intermediate dot result")),
          res_dot          (Kokkos::view_alloc(Kokkos::WithoutInitializing,       "res_dot is needed on both host and device"))
    {}

    //! Allow the user to interact with the preconditioner, *e.g.* to update its parameters between 2 subsequent calls to @ref apply.
    auto& get_preconditioner() { return preconditioner; }

    template <Kokkos::ExecutionSpace Exec>
    std::tuple<mag_t, decltype(base_t::Parameters::max_iters)> apply(const Exec& exec, const VectorType& sol, const base_t::Parameters& params) const
    {
        PLOG_INFO << "PCGGraph(apply): starting...";

        typename base_t::template spmv_handle_t<Exec> handle {};

#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::same_as<Exec, Kokkos::Cuda>) {
            algorithms::cg::check_cublas_uses_host_pointer_mode();
        }
#endif

        //! Pre-compute the residual.
        Kokkos::deep_copy(exec, res, rhs);
        KokkosSparse::spmv(exec, &handle, "N", -1., mat, sol, 1., res);

        //! Placeholder for the residual L2 norm.
        mag_t res_nrm2 = Kokkos::sqrt(Kokkos::abs(KokkosBlas::dot(exec, res, res)));
        if(res_nrm2 < params.tolerance) return {res_nrm2, 0};

        //! Pre-compute the residual of the preconditioned system.
        preconditioner.apply(exec, res_p, res);

        //! Compute the dot product of both residuals.
        KokkosBlas::dot(exec, res_res_p_dot_old, res_p, res);

        //! Direction of search is set to the residual of the preconditioned system.
        Kokkos::deep_copy(exec, dir, res_p);

        //! Create the graph.
        if(!graph.has_value())
        {
            const ::algorithms::cg::Region region_create_graph("PCGGraph - create graph");

            graph.emplace(Kokkos::Experimental::get_device_handle(exec));

            const auto tmp       = Kokkos::subview(scalars_on_device, 0);
            const auto alpha     = Kokkos::subview(scalars_on_device, 1);
            const auto alpha_neg = Kokkos::subview(scalars_on_device, 2);

            const auto alpha_spmv = ::algorithms::cg::spmv(graph->root_node(), &handle, "N", 1., mat, dir, 0., mat_dir);
            const auto alpha_dot  = ::algorithms::cg::dot(alpha_spmv, tmp, dir, mat_dir);

            const auto alpha_div = alpha_dot.then("alpha", Alpha<
                std::remove_cvref_t<decltype(alpha)>,
                std::remove_cvref_t<decltype(res_res_p_dot_old)>,
                std::remove_cvref_t<decltype(tmp)>
            >{.alpha = alpha, .alpha_neg = alpha_neg, .a = res_res_p_dot_old, .b = tmp});

            const auto update_sol = ::algorithms::cg::axpby(alpha_div, alpha,     dir,     1., sol);
            const auto update_res = ::algorithms::cg::axpby(alpha_div, alpha_neg, mat_dir, 1., res);

            ::algorithms::cg::dot(update_res, res_dot, res, res);

            const auto prec_apply = preconditioner.apply(update_res, res_p, res);

            const auto res_p_res_dot = ::algorithms::cg::dot(prec_apply, tmp, res_p, res);

            const auto beta_div = res_p_res_dot.then("beta", algorithms::cg::DivideAndSwap{.a = tmp, .b = res_res_p_dot_old});

            algorithms::cg::axpby(
                Kokkos::Experimental::when_all(update_sol, beta_div),
                1., res_p, tmp, dir
            );

            exec.fence("PCGGraph - fence before popping create graph region");
            region_create_graph.pop();

            //! Instantiate and upload the graph. This is important to ensure that the first submission directly runs.
            const ::algorithms::cg::Region region_instantiate_graph("PCGGraph - instantiate graph");

            graph->instantiate();

    #if defined(KOKKOS_ENABLE_CUDA)
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphUpload(graph->native_graph_exec(), exec.cuda_stream()));
    #endif

            exec.fence("PCGGraph - fence before popping instantiate graph region");
            region_instantiate_graph.pop();
        }

        const Kokkos::Profiling::ScopedRegion region("PCGGraph - loop");

        //! Loop until the norm of the residual is smaller than @p tol or the maximum number of iterations is reached.
        decltype(base_t::Parameters::max_iters) iter = 0;
        while(res_nrm2 > params.tolerance && iter < params.max_iters)
        {
            graph->submit(exec);
            exec.fence("fencing before evaluating convergence");
            res_nrm2 = Kokkos::sqrt(Kokkos::abs(res_dot()));
            PLOG_INFO << "PCGGraph(apply): iteration " << iter << ", res nrm2 " << res_nrm2;
            ++iter;
        }

        return {res_nrm2, iter};
    }
};

} // namespace algorithms::pcg

#endif // GRAPH_DISPATCHING_ALGORITHMS_PCG_GRAPH_HPP
