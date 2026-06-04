#ifndef GRAPH_DISPATCHING_ALGORITHMS_CG_KOKKOS_HPP
#define GRAPH_DISPATCHING_ALGORITHMS_CG_KOKKOS_HPP

#include "Kokkos_Graph.hpp"
#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"

#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_spmv.hpp"

#include "algorithms/cg/Base.hpp"
#include "algorithms/cg/Functors.hpp"
#include "algorithms/cg/Helpers.hpp"
#include "utils/pool.hpp"

namespace algorithms::cg
{

//! Conjugate gradient solver with @c Kokkos::Graph.
template <typename MatrixType, typename VectorType, bool UseHostNode>
struct CGGraph : public algorithms::cg::CGBase<MatrixType, VectorType>
{
    using base_t = algorithms::cg::CGBase<MatrixType, VectorType>;

    using typename base_t::dot_t;
    using typename base_t::mag_t;

    MatrixType mat;
    VectorType rhs;

    template <Kokkos::ExecutionSpace Exec, typename SizeType = typename Exec::size_type>
    std::tuple<mag_t, SizeType> apply(const utils::ExecutionSpacePool<Exec>& pool, const VectorType& sol, const typename base_t::Parameters& params) const
    {
        typename base_t::template spmv_handle_t<Exec> handle {};

#if defined(KOKKOS_ENABLE_CUDA)
        algorithms::cg::check_cublas_uses_host_pointer_mode();
#endif

        if(pool.size() < 1) Kokkos::abort("The pool size must be at least 1.");
        const auto& exec = pool.get(0);

        //! Pre-compute the residual.
        VectorType res(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual"), rhs.size());
        Kokkos::deep_copy(exec, res, rhs);
        KokkosSparse::spmv(exec, &handle, "N", -1., mat, sol, 1., res);

        //! Compute the residual norm, because it might already be all good.
        const Kokkos::View<dot_t, Kokkos::SharedHostPinnedSpace> res_dot_old("res_dot_old is needed on both host and device");
        res_dot_old() = KokkosBlas::dot(exec, res, res);

        //! Placeholder for the residual L2 norm.
        mag_t res_nrm2 = Kokkos::sqrt(Kokkos::abs(res_dot_old()));
        if(res_nrm2 < params.tolerance) return {res_nrm2, 0};

        //! Direction of search is set to the residual.
        const VectorType dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "dir"), rhs.size());
        Kokkos::deep_copy(exec, dir, res);

        //! Placeholder for the product of @ref mat with the direction of search.
        const VectorType mat_dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "mat * dir"), rhs.size());

        /**
         * For simplicity, the memory space is chosen as:
         *  - @c Kokkos::SharedHostPinnedSpace when using the host node
         *  - the device memory space otherwise.
         *
         * We must be extra careful when choosing the memory space for storing temporary variables.
         *
         * We've seen a huge performance drop when using the @c Kokkos::SharedHostPinnedSpace for variables
         * that need to be "sent" to SMs, especially when all SMs are occupied (at least on our @c BLACKWELL120 machine).
         *
         * Note that this choice therefore penalizes the host node version of this solver. However, as the performance of the
         * host node is always worse (by far) than the performance of the device then node, even for very small systems, we keep it
         * like that.
         *
         * Other strategies to mitigate the penalty of using the @c Kokkos::SharedHostPinnedSpace could be:
         *  - use @c Kokkos::SharedSpace
         *  - add memory copy nodes before and after the host node
         *
         * @note The read/write scenario for the host node is typically:
         *          - The CPU reads @c res_dot_old and @c tmp and writes @c alpha and @c alpha_neg.
         *          - The GPU reads @c alpha and @c alpha_neg in the kernels.
         *       It seems to fit into the 'write-combining' mode of page-locked host memory described in
         *       https://docs.nvidia.com/cuda/cuda-c-programming-guide/#write-combining-memory.
         */
        using memory_space_tmp_t = std::conditional_t<
            UseHostNode,
            Kokkos::SharedHostPinnedSpace,
            typename Exec::memory_space
        >;

        const Kokkos::View<dot_t[3], memory_space_tmp_t> scalars_on_device(Kokkos::view_alloc(
            Kokkos::WithoutInitializing, exec,
            "alpha is needed on both host and device"
        ));

        const auto tmp       = Kokkos::subview(scalars_on_device, 0);
        const auto alpha     = Kokkos::subview(scalars_on_device, 1);
        const auto alpha_neg = Kokkos::subview(scalars_on_device, 2);

        static_assert(std::remove_cvref_t<decltype(tmp)>::rank() == 0);

        //! Create the graph.
        const Region region_graph_definition("CGGraph - graph definition");
        Kokkos::Experimental::Graph graph(Kokkos::Experimental::get_device_handle(exec));

        //! Compute @c alpha.
        const auto alpha_spmv = ::algorithms::cg::spmv(graph.root_node(), &handle, "N", 1., mat, dir, 0., mat_dir);
        const auto alpha_dot  = ::algorithms::cg::dot(alpha_spmv, tmp, dir, mat_dir);

        const auto alpha_div = [&]() {
            if constexpr (UseHostNode) {
                return alpha_dot.then_host(
                    Kokkos::Experimental::node_props("alpha"),
                    [=]           { alpha() = res_dot_old() / tmp(); alpha_neg() = -alpha(); });
            } else {
                return alpha_dot.then(
                    Kokkos::Experimental::node_props("alpha", Kokkos::Experimental::get_device_handle(exec)),
                    KOKKOS_LAMBDA { alpha() = res_dot_old() / tmp(); alpha_neg() = -alpha(); });
            }
        }();

        /// Update the solution candidate and residual.
        const auto update_sol = ::algorithms::cg::axpby(alpha_div, alpha,     dir,     1., sol);
        const auto update_res = ::algorithms::cg::axpby(alpha_div, alpha_neg, mat_dir, 1., res);

        //! @todo At this point, we could already check the condition and exit with an if node and avoid the extra beta and search update if converged.
        const auto res_dot = ::algorithms::cg::dot(update_res, tmp, res, res);

        //! Compute @c beta.
        const auto beta_div = [&, das = algorithms::cg::DivideAndSwap{.a = tmp, .b = res_dot_old}]() mutable {
            if constexpr (UseHostNode) {
                return res_dot.then_host(
                    Kokkos::Experimental::node_props("beta"), std::move(das));
            } else {
                return res_dot.then(
                    Kokkos::Experimental::node_props("beta", Kokkos::Experimental::get_device_handle(exec)), std::move(das));
            }
        }();

        //! Update search direction, only once the solution has been updated.
        algorithms::cg::axpby(
            Kokkos::Experimental::when_all(update_sol, beta_div),
            1., res, tmp, dir
        );

        region_graph_definition.pop();

        //! Instantiate and upload the graph. This is important to ensure that the first submission directly runs.
        const Region region_graph_instantiation("CGGraph - graph instantiation");

        graph.instantiate();

        region_graph_instantiation.pop();

        //! Submit the first time outside the loop, for the same reason as in @ref benchmarks::graph::StraightLineBenchmark::run_graph.
        const Region region_graph_submit_0("CGGraph - submit 0");
        graph.submit(exec);
        exec.fence("fencing before evaluating convergence");
        res_nrm2 = Kokkos::sqrt(Kokkos::abs(res_dot_old()));

        region_graph_submit_0.pop();

        const Kokkos::Profiling::ScopedRegion region_graph_submit_r("CGGraph - submit r");

        //! Loop until the norm of the residual is smaller than @p tol or the maximum number of iterations is reached.
        SizeType iter = 1;
        while(res_nrm2 > params.tolerance && iter < params.max_iters)
        {
            graph.submit(exec);
            exec.fence("fencing before evaluating convergence");
            res_nrm2 = Kokkos::sqrt(Kokkos::abs(res_dot_old()));
            ++iter;
        }

        return {res_nrm2, iter};
    }
};

} // namespace algorithms::cg

#endif // #define GRAPH_DISPATCHING_ALGORITHMS_CG_KOKKOS_HPP
