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
 * Similar to @ref tests/cg/test_outlook.cpp, but adds @c Cuda memory nodes (manually since
 * it is not part of @c Kokkos yet).
 *
 * The test can be found in @ref cg/test_outlook.memcpy.cpp.
 */

namespace tests::cg
{
//! Direction of a memory copy operation.
enum class Direction
{
    DeviceToHost   = cudaMemcpyDeviceToHost,
    DeviceToDevice = cudaMemcpyDeviceToDevice
};

//! Add a @c memcpy graph node.
template <typename T, typename SrcType, typename DstType>
cudaGraphNode_t add_memcpy(const T& node, const DstType& dst, const SrcType& src, const Direction dir)
{
    cudaGraphNode_t* const native_node  = Kokkos::Impl::GraphAccess::get_node_ptr(node)->get_kernel().get_cuda_graph_node_ptr();
    const std::vector<cudaGraphNode_t> predecessors {*native_node};

    cudaGraph_t const* native_graph = Kokkos::Impl::GraphAccess::get_node_ptr(node)->get_kernel().get_cuda_graph_ptr();

    cudaGraphNode_t memcpy = nullptr;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphAddMemcpyNode1D(
        &memcpy, *native_graph,
        predecessors.data(), predecessors.size(),
        dst.data(),
        src.data(),
        src.size() * sizeof(typename SrcType::value_type),
        static_cast<cudaMemcpyKind>(dir)
    ));

    return memcpy;
}

/**
 * @brief Conjugate gradient solver with a @c Kokkos::Graph.
 *
 * References:
 *  - https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
 */
template <typename VectorType, typename MatrixType>
struct CGGraph
{
    using dot_t    = typename Kokkos::Details::InnerProductSpaceTraits<typename VectorType::non_const_value_type>::dot_type;
    using result_t = Kokkos::View<dot_t, typename VectorType::memory_space>; //! To store intermediate scalar results (norm, dot, and what not).

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
        result_t res_dot_old(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual dot - old"));
        result_t res_dot_new(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual dot - new"));

        KokkosBlas::dot(exec, res_dot_old, res, res);

        //! Direction of search is set to the residual.
        VectorType dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual"), rhs.size());
        Kokkos::deep_copy(exec, dir, res);

        //! Placeholder for the product of @ref mat with the direction of search.
        VectorType mat_dir(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "mat * dir"), rhs.size());

        //! Placeholder for the 'quadratic'.
        result_t quadratic(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "quadratic"));

        //! Placeholder for the 'alpha' and 'beta'.
        result_t alpha    (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "alpha"));
        result_t alpha_neg(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "alpha - negated"));
        result_t beta     (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "beta" ));

        //! Placeholder for the residual L2 norm (host variable because it's used in conditionals).
        dot_t res_nrm2 = 0.;

        //! Loop until the norm of the residual is larger than @p tol.
        Kokkos::deep_copy(exec, res_nrm2, res_dot_old);
        exec.fence("Wait for deep-copy into a host variable.");

        res_nrm2 = std::sqrt(res_nrm2);

        size_t iter = 0;

        //! Check for convergence even before creating the graph.
        if(res_nrm2 < tol || max_iter == 0) return std::tuple{res_nrm2, iter};

        //! Create the graph.
        auto root = Kokkos::Experimental::graph::create_graph(exec);

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

        //! Update residual L2-norm (host).
        add_memcpy(compute_res_dot_new, Kokkos::View<dot_t, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(&res_nrm2), res_dot_new, Direction::DeviceToHost);

        //! Compute @c beta.
        decltype(auto) beta_final = ::tests::cg::scalar_div(std::move(compute_res_dot_new), beta, res_dot_new, res_dot_old);

        //! Update residual dot on device.
        add_memcpy(beta_final, res_dot_old, res_dot_new, Direction::DeviceToDevice);

        //! Update search direction.
        decltype(auto) update_dir = ::tests::cg::axpby(std::move(beta_final), 1., res, beta, dir);

        //! Loop until convergence.
        while(res_nrm2 > tol && iter < max_iter)
        {
            std::cout << "> Iteration " << iter << ": residual norm is " << res_nrm2 << std::endl;
            Kokkos::Profiling::ScopedRegion region("iter-" + std::to_string(iter));

            Kokkos::Experimental::graph::submit(exec, update_dir);

            exec.fence("Wait for deep-copy into a host variable.");

            ++iter;
        }

        return std::tuple{res_nrm2, iter};
    }
};

//! @test Use @ref CGGraph to solve the 2-by-2 system created by @ref TwoByTwo.
TEST(CGGraph, 2x2)
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

    using twobytwo_t = TwoByTwo<double, Kokkos::Device<execution_space, memory_space>>;

    const execution_space exec {};

    twobytwo_t sys(exec);

    CGGraph solver{.rhs = std::move(sys.rhs), .mat = std::move(sys.matrix)};

    const auto [res_nrm2, num_iters] = solver.apply(exec, sys.guess, 1.e-5, 5);

    ASSERT_LT(res_nrm2,  1.e-5);
    ASSERT_LT(num_iters, 3);

    const auto sol_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sys.guess);

    ASSERT_DOUBLE_EQ(sol_h(0), twobytwo_t::sol_0);
    ASSERT_DOUBLE_EQ(sol_h(1), twobytwo_t::sol_1);
}

} // namespace tests::cg
