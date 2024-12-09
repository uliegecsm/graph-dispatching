#include "gtest/gtest.h"

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "Kokkos_Profiling_ScopedRegion.hpp"
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
 * The test can be found in @ref cg/test_outlook.cpp.
 */

namespace tests::cg
{

//! Conjugate gradient solver with a @c Kokkos::Graph.
template <typename VectorType, typename MatrixType>
struct CGGraph : public ConjugateGradientSolverBase<VectorType, MatrixType>
{
    using base_t = ConjugateGradientSolverBase<VectorType, MatrixType>;

    using typename base_t::device_t;
    using typename base_t::device_um_t;
    using typename base_t::dot_t;
    using typename base_t::pinned_t;
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
        pinned_t res_dot_old(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual dot - old"));
        pinned_t res_dot_new(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "residual dot - new"));

        KokkosBlas::dot(exec, res_dot_old, res, res);

        //! Check for convergence even before creating the graph.
        exec.fence("Check for convergence on host during initialization phase.");
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

        //! Compute @c beta.
        decltype(auto) beta_final = ::tests::cg::scalar_div(std::move(compute_res_dot_new), beta, res_dot_new, res_dot_old);

        //! Update search direction.
        decltype(auto) update_dir = ::tests::cg::axpby(std::move(beta_final), 1., res, beta, dir);

        //! Loop until convergence.
        size_t iter = 0;
        while(res_nrm2 > tol && iter < max_iter)
        {
            Kokkos::Profiling::ScopedRegion region("iter-" + std::to_string(iter));

            Kokkos::Experimental::graph::submit(exec, update_dir);

            exec.fence("Wait for graph submission to finish before evaluating convergence.");

            res_dot_old() = res_dot_new();
            ++iter;
            res_nrm2 = std::sqrt(res_dot_old());
        }

        return {res_nrm2, iter};
    }
};

using CGGraphTest = NbyNSolverTest<CGGraph<
    NbyNSolverTestHelper::initializer_t::values_t,
    NbyNSolverTestHelper::initializer_t::matrix_t
>>;

TEST_F(CGGraphTest, 10x10)
{
    this->run(10);
}

} // namespace tests::cg
