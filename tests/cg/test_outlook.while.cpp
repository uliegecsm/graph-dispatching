#include "gtest/gtest.h"

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"
#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_spmv.hpp"

#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with a @c Kokkos::Graph and @c Cuda @c memcpy nodes
 * -----------------------------------------------------------------------------
 *
 * Similar to @ref tests/cg/test_outlook.cpp, but the while loop is a @c Cuda node.
 *
 * The test can be found in @ref cg/test_outlook.while.cpp.
 */

namespace tests::cg
{
#if defined(ADD_CONVERGENCE_WITH_THEN)
//! @todo We get a fresh copy each time of @c functor, look at global launch instead ?
template <typename ConvergenceType>
__global__ void convergence(ConvergenceType *const functor, cudaGraphConditionalHandle handle) {
    if(!functor->operator()())
        cudaGraphSetConditional(handle, false);
}
#endif

//! Update the @c beta coefficient and evaluate the convergence criterion.
template <typename ViewType, typename CounterType, typename ResType> requires (ViewType::rank() == 0 && CounterType::rank() == 0)
struct Convergence
{
    CounterType iter;
    typename CounterType::value_type max_iter = 0;
    typename ViewType::value_type tol = 0.;

    ResType res_dot;
    ViewType beta;

    /// @note @c cudaGraphSetConditional is a pure device function.
    ///       Therefore, we cannot decorate our call operator with @ref KOKKOS_FUNCTION.
    __device__
    bool operator()()
    {
        ++iter();
        beta() = res_dot(1) / res_dot(0);
        res_dot(0) = res_dot(1);
        return ! (std::sqrt(res_dot(1)) < tol || iter() > max_iter);
    }
};

/**
 * @brief Conjugate gradient solver with a @c Kokkos::Graph.
 *
 * References:
 *  - https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
 */
template <typename VectorType, typename MatrixType>
struct CGGraphWhile : public ConjugateGradientSolverBase<VectorType, MatrixType>
{
    using base_t = ConjugateGradientSolverBase<VectorType, MatrixType>;

    using typename base_t::counter_t;
    using typename base_t::device_t;
    using typename base_t::device_um_t;
    using typename base_t::dot_t;
    using typename base_t::res_dot_t;

    VectorType rhs;
    MatrixType mat;

    template <typename Exec>
    std::tuple<dot_t, size_t> apply(const Exec& exec, const VectorType& sol, const VectorType::value_type tol, const size_t max_iter) const
    {
        using spmv_handle_t = KokkosSparse::SPMVHandle<typename VectorType::memory_space, MatrixType, VectorType, VectorType>;
        spmv_handle_t handle {};

        //! Pre-compute the residual.
        VectorType res(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual"), rhs.size());
        Kokkos::deep_copy(exec, res, rhs);
        KokkosSparse::spmv(exec, &handle, "N", -1., mat, sol, 1., res);

        //! Placeholder for the dot product of the residual with itself.
        res_dot_t res_dot(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual dot - old at 0 and new at 1"));
        device_um_t res_dot_old(res_dot.data());
        device_um_t res_dot_new(res_dot.data() + 1);

        KokkosBlas::dot(exec, res_dot_old, res, res);

        //! Check for convergence even before creating the graph.
        dot_t res_nrm2 = 0.;
        Kokkos::deep_copy(exec, res_nrm2, res_dot_old);
        exec.fence("Check for convergence on host during initialization phase.");
        if(res_nrm2 = std::sqrt(res_nrm2); res_nrm2 < tol) return {res_nrm2, 0};

        //! Direction of search is set to the residual.
        VectorType dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual"), rhs.size());
        Kokkos::deep_copy(exec, dir, res);

        //! Placeholder for the product of @ref mat with the direction of search.
        VectorType mat_dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "mat * dir"), rhs.size());

        //! Placeholder for the 'quadratic'.
        device_t quadratic(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "quadratic"));

        //! Placeholder for the 'alpha' and 'beta'.
        device_t alpha    (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "alpha"));
        device_t alpha_neg(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "alpha - negated"));
        device_t beta     (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "beta" ));

        //! Placeholder for the iteration count.
        counter_t iter(Kokkos::view_alloc(exec, "iterations"));

        //! Create the main graph.
        Kokkos::Timer elapsed_before_launch;
        cudaGraph_t graph = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphCreate(&graph, 0));

        cudaGraphConditionalHandle conditional_handle;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphConditionalHandleCreate(&conditional_handle, graph, 1, cudaGraphCondAssignDefault));

        cudaGraphNodeParams conditional_params = {};
        conditional_params.type                = cudaGraphNodeTypeConditional;
        conditional_params.conditional.handle  = conditional_handle;
        conditional_params.conditional.type    = cudaGraphCondTypeWhile;
        conditional_params.conditional.size    = 1;
        cudaGraphNode_t conditional_node = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphAddNode(&conditional_node, graph, nullptr, 0, &conditional_params));

        //! Create the graph.
        // auto root = Kokkos::Experimental::graph::create_graph(exec);
        auto root = Kokkos::Experimental::graph::details::ChainHandler<Exec>(
            Kokkos::Impl::GraphAccess::construct_graph_from_native(
                exec,
                conditional_params.conditional.phGraph_out[0]
            )
        );

        //! Compute @c alpha.
        decltype(auto) alpha_spmv = ::tests::cg::spmv(std::move(root), handle, 1., mat, dir, 0., mat_dir);

        decltype(auto) alpha_dot = ::tests::cg::dot(std::move(alpha_spmv), quadratic, dir, mat_dir);

        decltype(auto) alpha_final = ::tests::cg::scalar_div_and_neg(std::move(alpha_dot), alpha, alpha_neg, res_dot_old, quadratic);

        //! @todo We need to expose our @c operator|, otherwise the compiler can't find a match.
        using Kokkos::Experimental::graph::details::operator|;
        decltype(auto) alpha_split = std::move(alpha_final) | Kokkos::Experimental::graph::split();

        //! Update the solution candidate.
        decltype(auto) update_sol = ::tests::cg::axpby(alpha_split, alpha, dir, 1., sol);

        //! Update the residual.
        decltype(auto) update_res = ::tests::cg::axpby(std::move(alpha_split), alpha_neg, mat_dir, 1., res);

        //! At this point, we could already check the condition and exit, but we don't have conditional nodes yet.
        decltype(auto) compute_res_dot_new = ::tests::cg::dot(std::move(update_res), res_dot_new, res, res);

        //! Compute @c beta and convergence check.
        using convergence_t = Convergence<device_t, counter_t, res_dot_t>;
        convergence_t convergence_functor {
            .iter        = iter,
            .max_iter    = max_iter,
            .tol         = tol,
            .res_dot     = res_dot,
            .beta        = beta
        };
#if defined(ADD_CONVERGENCE_WITH_THEN)
        Kokkos::View<convergence_t, typename VectorType::memory_space> convergence_functor_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "convergence functor"));
        Kokkos::deep_copy(exec, convergence_functor_d, std::move(convergence_to_be_moved));
        decltype(auto) beta_and_conv = std::move(compute_res_dot_new) | Kokkos::Experimental::graph::then<Exec>(
            [convergence_functor_d = std::move(convergence_functor_d), conditional_handle](const Exec& exec) {
                convergence<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, exec.cuda_stream()>>>(convergence_functor_d.data(), conditional_handle);
            });
#else
        decltype(auto) beta_and_conv = std::move(compute_res_dot_new) | Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy<Exec>(0, 1),
            KOKKOS_LAMBDA<typename T>(const T) {
                if(!const_cast<convergence_t&>(convergence_functor).operator()()) cudaGraphSetConditional(conditional_handle, false);
            }
        );
#endif

        //! Update search direction.
        decltype(auto) update_dir = ::tests::cg::axpby(std::move(beta_and_conv), 1., res, beta, dir);

        //! Add @c Kokkos graph as child graph node to the output of the @c while node.
        // cudaGraphNode_t subgraph_node = nullptr;
        // KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphAddChildGraphNode(
        //     &subgraph_node, 
        //     conditional_params.conditional.phGraph_out[0],
        //     nullptr, 0,
        //     *Kokkos::Impl::GraphAccess::get_node_ptr(update_dir)->get_kernel().get_cuda_graph_ptr()
        // ));

        //! Create the executable graph and submit once. It will converge during the first submission since the @c while is embedded.
        cudaGraphExec_t graph_exec = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        std::cout << "> Elapsed after call to instantiate: " << elapsed_before_launch.seconds() << " seconds." << std::endl;

        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphLaunch(graph_exec, exec.cuda_stream()));

        typename counter_t::value_type iter_h = 0;
        Kokkos::deep_copy(exec, iter_h, iter);

        typename device_t::value_type res_dot_h = 0.;
        Kokkos::deep_copy(exec, res_dot_h, res_dot_old);

        exec.fence("Wait for scalars.");

        //! Destroy graphs.
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphExecDestroy(graph_exec));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphDestroy    (graph));

        return std::tuple{std::sqrt(res_dot_h), iter_h};
    }
};

ADD_CG_TEST(CGGraphWhile)

} // namespace tests::cg
