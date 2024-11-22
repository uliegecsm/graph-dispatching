#include "gtest/gtest.h"

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_spmv.hpp"

#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with a @c Kokkos::Graph
 * -------------------------------------------------
 *
 * Implement a portable conjugate gradient solver using @c Kokkos::Graph.
 *
 * The test can be found in @ref capture/test_outlook.cpp.
 */

namespace tests::cg
{

/**
 * @brief Perform SPMV, graph-compatible.
 *
 * The implementation relies on the @c KokkosSparse kernel. Inspired by
 * https://github.com/trilinos/Trilinos/blob/62023ad68e09a2972240971c40be34465010d6f3/packages/kokkos-kernels/perf_test/sparse/KokkosSparse_spmv_struct_tuning.cpp#L191.
 *
 * @warning It has not been tuned at all.
 */
template <typename Exec, typename Handle, typename Alpha, typename AMatrix, typename XVector, typename Beta, typename YVector>
decltype(auto) spmv(Exec&& exec, const Handle&, const Alpha& alpha, const AMatrix& mat, const XVector& vec_x, const Beta& beta, const YVector& vec_y)
{
    using execution_space = typename std::remove_cvref_t<Exec>::execution_space;

    KokkosSparse::Impl::SPMV_Functor<
        execution_space,
        AMatrix,
        XVector, YVector,
        1     /* dobeta */,
        false /* conjugate */
    > functor(alpha, mat, vec_x, beta, vec_y, /* rows_per_team */ 1);

    return std::forward<Exec>(exec) | Kokkos::Experimental::graph::parallel_for(
        "SPMV",
        Kokkos::TeamPolicy<execution_space>(1, Kokkos::AUTO),
        std::move(functor)
    );
}

//! Dot product, graph-compatible.
template <typename Exec, typename Result, typename ViewX, typename ViewY>
decltype(auto) dot(Exec&& exec, Result&& result, const ViewX& vec_x, const ViewY& vec_y)
{
    using execution_space = typename std::remove_cvref_t<Exec>::execution_space;

    return std::forward<Exec>(exec) | Kokkos::Experimental::graph::parallel_reduce(
        "DOT",
        Kokkos::RangePolicy<execution_space>(0, vec_x.size()),
        KOKKOS_LAMBDA(const typename execution_space::size_type index, typename ViewX::non_const_value_type& current) {
            current += vec_x(index) * vec_y(index);
        },
        std::forward<Result>(result)
    );
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
        if(res_nrm2 > tol && iter < max_iter) return std::tuple{res_nrm2, iter};

        //! Create the graph.
        auto root = Kokkos::Experimental::graph::create_graph(exec);

        //! Compute @c alpha.
        decltype(auto) alpha_spmv = spmv(std::move(root), handle, 1., mat, dir, 0., mat_dir);

        decltype(auto) alpha_dot = dot(std::move(alpha_spmv), quadratic, dir, mat_dir);

        decltype(auto) alpha_final = scalar_div_and_neg(std::move(alpha_dot), alpha, alpha_neg, res_dot_old, quadratic);

        //! Loop towards convergence.
        while(res_nrm2 > tol && iter < max_iter)
        {
            std::cout << "> Iteration " << iter << ": residual norm is " << res_nrm2 << std::endl;

            Kokkos::Experimental::graph::submit(exec, root);

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
