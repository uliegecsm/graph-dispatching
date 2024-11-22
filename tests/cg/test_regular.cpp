#include "gtest/gtest.h"

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "KokkosBlas1_update.hpp"
#include "KokkosSparse_spmv.hpp"

#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with regular @c Kokkos execution space instances
 * --------------------------------------------------------------------------
 *
 * Implement a portable conjugate gradient solver without using @c Kokkos::Graph.
 *
 * The test can be found in @ref capture/test_regular.cpp.
 */

namespace tests::cg
{
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
 * @brief Conjugate gradient solver with regular @c Kokkos execution space instances.
 *
 * References:
 *  - https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
 */
template <typename VectorType, typename MatrixType>
struct CGRegular
{
    using dot_t    = typename Kokkos::Details::InnerProductSpaceTraits<typename VectorType::non_const_value_type>::dot_type;
    using result_t = Kokkos::View<dot_t, typename VectorType::memory_space>; //! To store intermediate scalar results (norm, dot, and what not).

    VectorType rhs;
    MatrixType mat;

    //! We explicitly don't partition @p exec.
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

        while(res_nrm2 > tol && iter < max_iter)
        {
            std::cout << "> Iteration " << iter << ": residual norm is " << res_nrm2 << std::endl;

            //! Compute @c alpha.
            KokkosSparse::spmv(exec, &handle, "N", 1., mat, dir, 0., mat_dir);

            KokkosBlas::dot(exec, quadratic, dir, mat_dir);

            scalar_div_and_neg(exec, alpha, alpha_neg, res_dot_old, quadratic);

            //! Update the solution candidate.
            axpby(exec, alpha, dir, 1., sol);

            //! Update the residual.
            axpby(exec, alpha_neg, mat_dir, 1., res);

            //! At this point, we can already check the condition and exit.
            KokkosBlas::dot(exec, res_dot_new, res, res);

            Kokkos::deep_copy(exec, res_nrm2, res_dot_new);
            exec.fence("Wait for deep-copy into a host variable.");

            if((res_nrm2 = std::sqrt(res_nrm2)) > tol)
            {
                //! Compute @c beta.
                scalar_div(exec, beta, res_dot_new, res_dot_old);

                //! Update search direction.
                axpby(exec, 1., res, beta, dir);

                Kokkos::deep_copy(exec, res_dot_old, res_dot_new);

                ++iter;
            }
        }

        return std::tuple{res_nrm2, iter};
    }
};

//! @test Use @ref CGRegular to solve the 2-by-2 system created by @ref TwoByTwo.
TEST(CGRegular, 2x2)
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

    using twobytwo_t = TwoByTwo<double, Kokkos::Device<execution_space, memory_space>>;

    const execution_space exec {};

    twobytwo_t sys(exec);

    CGRegular solver{.rhs = std::move(sys.rhs), .mat = std::move(sys.matrix)};

    const auto [res_nrm2, num_iters] = solver.apply(exec, sys.guess, 1.e-5, 5);

    ASSERT_LT(res_nrm2,  1.e-5);
    ASSERT_LT(num_iters, 3);

    const auto sol_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sys.guess);

    ASSERT_DOUBLE_EQ(sol_h(0), twobytwo_t::sol_0);
    ASSERT_DOUBLE_EQ(sol_h(1), twobytwo_t::sol_1);
}

} // namespace tests::cg
