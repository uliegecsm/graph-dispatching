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
//! @todo We get a fresh copy each time of @c functor, look at global launch instead ?
template <typename ConvergenceType>
__global__ void convergence(ConvergenceType functor, cudaGraphConditionalHandle handle) {
    if(!functor())
        cudaGraphSetConditional(handle, false);
}

/**
 * @brief Update the @c beta coefficient and evaluate the convergence criterion.
 *
 * @todo Should we use a global launch ? (one object so states can be scalars)
 *       Or local launch ? (we get a new one each time so states must be views)
 */
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
    using typename base_t::dot_t;

    /// We store residual norm of current and previous iteration side by side to use less load instructions.
    /// It is also a single allocation instead of 2, thereby reducing the @c Cuda API CPU overhead.
    using res_dot_t = Kokkos::View<dot_t[2], typename VectorType::memory_space>;

    //! Useful type for getting an unmanaged subview of @ref reso_dot_t.
    using device_um_t = Kokkos::View<dot_t, typename VectorType::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    VectorType rhs;
    MatrixType mat;

    //! Create a @c Kokkos::Graph to iterate towards the solution.
    template <typename Exec>
    auto apply(const Exec& exec, const VectorType& sol, const VectorType::value_type tol, const size_t max_iter) const
    {
        using spmv_handle_t = KokkosSparse::SPMVHandle<typename VectorType::memory_space, MatrixType, VectorType, VectorType>;
        spmv_handle_t handle {};

        //! Pre-compute the residual.
        VectorType res(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual"), rhs.size());
        Kokkos::deep_copy(exec, res, rhs);
        KokkosSparse::spmv(exec, &handle, "N", -1., mat, sol, 1., res);

        //! Placeholder for the dot product of the residual with itself.
        res_dot_t res_dot(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual dot - old at 0 and new at 1"));

        KokkosBlas::dot(exec, device_um_t(res_dot.data()), res, res);

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
        counter_t iter(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "iterations"));

        //! Check for convergence even before creating the graph.
        // if(res_nrm2 < tol || max_iter == 0) return std::tuple{res_nrm2, iter};

        //! Create the graph.
        auto root = Kokkos::Experimental::graph::create_graph(exec);

        //! Compute @c alpha.
        decltype(auto) alpha_spmv = ::tests::cg::spmv(std::move(root), handle, 1., mat, dir, 0., mat_dir);

        decltype(auto) alpha_dot = ::tests::cg::dot(std::move(alpha_spmv), quadratic, dir, mat_dir);

        decltype(auto) alpha_final = ::tests::cg::scalar_div_and_neg(std::move(alpha_dot), alpha, alpha_neg, device_um_t(res_dot.data()), quadratic);

        //! @todo We need to expose our @c operator|, otherwise the compiler can't find a match.
        using Kokkos::Experimental::graph::details::operator|;
        decltype(auto) alpha_split = std::move(alpha_final) | Kokkos::Experimental::graph::split();

        //! Update the solution candidate.
        decltype(auto) update_sol = ::tests::cg::axpby(alpha_split, alpha, dir, 1., sol);

        //! Update the residual.
        decltype(auto) update_res = ::tests::cg::axpby(std::move(alpha_split), alpha_neg, mat_dir, 1., res);

        //! At this point, we could already check the condition and exit, but we don't have conditional nodes yet.
        decltype(auto) compute_res_dot_new = ::tests::cg::dot(std::move(update_res), device_um_t(res_dot.data() + 1), res, res);

        //! Create the main graph. The @c Kokkos graph will be added as a child graph node.
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

        //! Compute @c beta and convergence check.
        Convergence<device_t, counter_t, res_dot_t> convergence_functor {
            .iter        = iter,
            .max_iter    = max_iter,
            .tol         = tol,
            .res_dot     = res_dot,
            .beta        = beta
        };
        decltype(auto) beta_and_conv = std::move(compute_res_dot_new) | Kokkos::Experimental::graph::then<Exec>(
            [convergence_functor = std::move(convergence_functor), conditional_handle](const Exec& exec) {
                convergence<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, exec.cuda_stream()>>>(convergence_functor, conditional_handle);
            });

        //! Update search direction.
        decltype(auto) update_dir = ::tests::cg::axpby(std::move(beta_and_conv), 1., res, beta, dir);

        //! Add @c Kokkos graph as child graph node to the output of the @c while node.
        cudaGraphNode_t subgraph_node = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphAddChildGraphNode(
            &subgraph_node, 
            conditional_params.conditional.phGraph_out[0],
            nullptr, 0,
            *Kokkos::Impl::GraphAccess::get_node_ptr(update_dir)->get_kernel().get_cuda_graph_ptr()
        ));

        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphDebugDotPrint(graph, "test_outlook.while.dot", cudaGraphDebugDotFlagsVerbose));

        //! Create the executable graph and submit once. It will converge during the first submission since the @c while is embedded.
        cudaGraphExec_t graph_exec = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphLaunch(graph_exec, exec.cuda_stream()));

        exec.fence();

        //! Destroy graphs.
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphExecDestroy(graph_exec));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphDestroy    (graph));

        typename counter_t::value_type iter_h = 0;
        Kokkos::deep_copy(exec, iter_h, iter);

        typename device_t::value_type res_dot_h = 0.;
        Kokkos::deep_copy(exec, res_dot_h, device_um_t(res_dot.data()));

        exec.fence("wait for scalars");

        return std::tuple{std::sqrt(res_dot_h), iter_h};
    }
};

using CGGraphWhileTest = NbyNSolverTest<CGGraphWhile<
    NbyNSolverTestHelper::initializer_t::values_t,
    NbyNSolverTestHelper::initializer_t::matrix_t
>>;

TEST_F(CGGraphWhileTest, 10x10)
{
    this->run(10);
}

} // namespace tests::cg
