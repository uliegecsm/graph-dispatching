#include "gtest/gtest.h"

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"
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
 * The test can be found in @ref cg/test_regular.cpp.
 */

namespace tests::cg
{
/**
 * Wrapper around @c KokkosBlas::axpby to ensure fencing is done, otherwise
 * asynchronicity issues may arise. See also https://github.com/kokkos/kokkos-kernels/issues/2434.
 */
template <typename Exec, typename... Args>
void axpby_fence(const Exec& exec, Args&&... args)
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
struct CGRegular : public ConjugateGradientSolverBase<VectorType, MatrixType>
{
    using base_t = ConjugateGradientSolverBase<VectorType, MatrixType>;

    using typename base_t::device_t;
    using typename base_t::device_um_t;
    using typename base_t::pinned_um_t;
    using typename base_t::dot_t;
    using typename base_t::pinned_t;
    using typename base_t::res_dot_pinned_t;
    using typename base_t::res_dot_t;

    VectorType rhs;
    MatrixType mat;

    //! We explicitly don't partition @p exec.
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
        res_dot_pinned_t res_dot(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual dot - old at 0 and new at 1"));
        pinned_um_t res_dot_old(res_dot.data());
        pinned_um_t res_dot_new(res_dot.data() + 1);

        KokkosBlas::dot(exec, res_dot_old, res, res);

        //! Placeholder for the residual L2 norm (host variable because it's used in conditionals).
        exec.fence("Wait before reading residual dot product with itself.");
        dot_t res_nrm2 = std::sqrt(res_dot_old());
        if(res_nrm2 < tol) return {res_nrm2, 0};

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

        //! Loop until the norm of the residual is larger than @p tol.
        size_t iter = 0;
        while(res_nrm2 > tol && iter < max_iter)
        {
            std::cout << "> Iteration " << iter << ": residual norm is " << res_nrm2 << std::endl;
            Kokkos::Profiling::ScopedRegion region("iter-" + std::to_string(iter));

            //! Compute @c alpha.
            KokkosSparse::spmv(exec, &handle, "N", 1., mat, dir, 0., mat_dir);

            KokkosBlas::dot(exec, quadratic, dir, mat_dir);

            scalar_div_and_neg(exec, alpha, alpha_neg, res_dot_old, quadratic);

            //! Update the solution candidate.
            axpby_fence(exec, alpha, dir, 1., sol);

            //! Update the residual.
            axpby_fence(exec, alpha_neg, mat_dir, 1., res);

            //! At this point, we can already check the condition and exit.
            KokkosBlas::dot(exec, res_dot_new, res, res);

            exec.fence("Wait before reading residual dot product with itself.");
            if((res_nrm2 = std::sqrt(res_dot_new())) > tol)
            {
                //! Compute @c beta.
                scalar_div(exec, beta, res_dot_new, res_dot_old);

                //! Update search direction.
                axpby_fence(exec, 1., res, beta, dir);

                res_dot_old() = res_dot_new();
            }

            ++iter;
        }

        return std::tuple{res_nrm2, iter};
    }
};

ADD_CG_TEST(CGRegular)

} // namespace tests::cg
