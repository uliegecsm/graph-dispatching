#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with @c Kokkos::Graph
 * -----------------------------------------------
 *
 * Implement a portable conjugate gradient solver embedded in a @c Kokkos::Graph.
 *
 * The test can be found in @ref capture/test_outlook.cpp.
 */

namespace tests::cg
{
/**
 * @brief Scalar division on device plus negation.
 *
 * References:
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu#L103C1-L108C2
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu#L96
 */
template <typename Exec, typename T, typename U, typename V>
void scalar_div_and_neg(const Exec& exec, const T& out, const T& out_neg, const U& x, const V& y)
{
    Kokkos::parallel_for(
        "scalar division + negate",
        Kokkos::RangePolicy(exec, 0, 1),
        KOKKOS_LAMBDA(const int){ out() = x() / y(); out_neg() = - out(); }
    );
}

/**
 * @brief Scalar division on device.
 *
 * References:
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu#L103C1-L108C2
 */
template <typename Exec, typename T, typename U, typename V>
void scalar_div(const Exec& exec, const T& out, const U& x, const V& y)
{
    Kokkos::parallel_for(
        "scalar division",
        Kokkos::RangePolicy(exec, 0, 1),
        KOKKOS_LAMBDA(const int){ out() = x() / y(); }
    );
}

/**
 * Wrapper around @c KokkosBlas::axpby to ensure fencing is done, otherwise
 * asynchronicity issues may arise. See also https://github.com/kokkos/kokkos-kernels/issues/2434.
 */
template <typename Exec, typename... Args>
void axpby(const Exec& exec, Args&&... args)
{
    exec.fence("Fencing before calling 'KokkosBlas::axpby'.");
    KokkosBlas::axpby(exec, std::forward<Args>(args)...);
}

/**
 * @brief Conjugate gradient solver.
 *
 * References:
 *  - https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
 */
template <typename VectorType, typename MatrixType>
struct CG
{
    VectorType rhs;
    MatrixType mat;

    using dot_t = typename Kokkos::Details::InnerProductSpaceTraits<typename VectorType::non_const_value_type>::dot_type;

    using result_t = Kokkos::View<dot_t, Kokkos::HostSpace>; //! To store intermediate scalar results (norm, dot). @note It should move to device or shared space.

    /**
     * @warning For now, we cannot portably add conditionals to the graph, so we have to submit the graph
     *          from within this function. Therefore, we cannot yet support both execution space and graph mode.
     */
    template <typename Exec>
    auto apply(const Exec& exec, const VectorType& sol, const VectorType::value_type tol, const size_t max_iter) const
    {
        //! Initialize the graph.
        // decltype(auto) root = Kokkos::Experimental::graph::create_graph(exec);

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

        //! Placeholder for the residual L2 norm.
        dot_t res_nrm2;

        //! Loop until the norm of the residual is larger than @p tol.
        exec.fence("Wait for the residual dot product to be computed.");
        size_t iter = 0;
        res_nrm2 = std::sqrt(res_dot_old());

        while(res_nrm2 > tol && iter < max_iter)
        {
            std::cout << "> Iteration " << iter << ": residual norm is " << res_nrm2 << std::endl;

            //! Compute @c alpha.
            KokkosBlas::dot(exec, res_dot_old, res, res);

            KokkosSparse::spmv(exec, &handle, "N", 1., mat, dir, 0., mat_dir);

            KokkosBlas::dot(exec, quadratic, dir, mat_dir);

            scalar_div_and_neg(exec, alpha, alpha_neg, res_dot_old, quadratic);

            //! Update the solution candidate.
            axpby(exec, alpha, dir, 1., sol);

            //! Update the residual.
            axpby(exec, alpha_neg, mat_dir, 1., res);

            //! At this point, we can already check the condition and exit.
            KokkosBlas::dot(exec, res_dot_new, res, res);

            exec.fence("Wait for rank-0 views to be ready.");

            if((res_nrm2 = std::sqrt(res_dot_new())) > tol)
            {
                //! Compute @c beta.
                scalar_div(exec, beta, res_dot_new, res_dot_old);

                //! Update search direction.
                axpby(exec, 1., res, beta, dir);

                ++iter;
            }
        }

        return res_nrm2;
    }
};

/**
 * @test Use @ref CG to solve a 2-by-2 system.
 *
 * The system is
 * \f[
 *   \begin{bmatrix}
 *      4 & 1 \\
 *      1 & 3
 *   \end{bmatrix}
 *   \begin{bmatrix}
 *      x_1 \\
 *      x_2
 *   \end{bmatrix}
 *   =
 *   \begin{bmatrix}
 *      1 \\
 *      2
 *   \end{bmatrix}
 * \f]
 * and the solution is
 * \f[
 *   \begin{bmatrix}
 *      \cfrac{1}{11} \\
 *      \cfrac{7}{11}
 *   \end{bmatrix}
 * \f]. We will use the following guess
 * \f[
 *   \begin{bmatrix}
 *      2 \\
 *      1
 *   \end{bmatrix}
 * \f].
 */
TEST(CGTest, 2x2)
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

    using matrix_t = KokkosSparse::CrsMatrix<
        /* scalar type */ double,
        /* index type  */ int,
        /* device      */ Kokkos::Device<execution_space, memory_space>
    >;
    using graph_t   = typename matrix_t::staticcrsgraph_type;
    using row_map_t = typename graph_t::row_map_type;
    using entries_t = typename graph_t::entries_type;
    using values_t  = typename matrix_t::values_type;

    const execution_space exec {};

    typename row_map_t::non_const_type row_map(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "row map" ), 3);
    typename entries_t::non_const_type entries(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "entries" ), 4);
    typename values_t::non_const_type  values (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "value"   ), 4);
    typename values_t::non_const_type  rhs    (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rhs"     ), 2);
    typename values_t::non_const_type  sol    (Kokkos::view_alloc(                             exec, "solution"), 2);

    Kokkos::parallel_for(
        "Set matrix and RHS values",
        Kokkos::RangePolicy(exec, 0, 1),
        KOKKOS_LAMBDA(const int)
        {
            row_map(0) = 0;
            row_map(1) = 2;
            row_map(2) = 4;

            entries(0) = 0; entries(1) = 1;
            entries(2) = 0; entries(3) = 1;

            values(0) = 4.; values(1) = 1.;
            values(2) = 1.; values(3) = 3.;

            rhs(0) = 1.; rhs(1) = 2.;

            sol(0) = 2.; sol(1) = 1.;
        }
    );

    matrix_t mat("matrix", 2, 2, 4, std::move(values), std::move(row_map), std::move(entries));

    CG solver{.rhs = std::move(rhs), .mat = std::move(mat)};

    const auto res_nrm2 = solver.apply(exec, sol, 1.e-5, 3);

    ASSERT_LT(res_nrm2, 1.e-5);

    const auto sol_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol);

    ASSERT_DOUBLE_EQ(sol_h(0), 1./11.);
    ASSERT_DOUBLE_EQ(sol_h(1), 7./11.);
}

} // namespace tests::cg
