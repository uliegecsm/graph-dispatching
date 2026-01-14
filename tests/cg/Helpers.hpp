#ifndef GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP

#include "Kokkos_InnerProductSpaceTraits.hpp"

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"

#include "kokkos-utils/concepts/ExecutionSpace.hpp"
#include "kokkos-utils/concepts/MemorySpace.hpp"
#include "kokkos-utils/timer/Timer.hpp"

namespace tests::cg
{
/**
 * @brief Create a N-by-N tri-diagonal symmetric system to test solvers.
 *
 * Inspired by a 1D Laplacian FEM problem with a Dirichlet BC on the left and
 * a non-homogenous Neumann on the right.
 *
 * The right-hand side value is scaled by the number of rows
 * to keep the solution values in a relatively small range.
 *
 * @verbatim
 * 1  0                     0
 * 0  2 -1                  0
 *       *                  *
 *           *              *
 *              *           *
 *             -1  2 -1     0
 *                -1  1     (2 + 2i) / size
 * @endverbatim
 *
 * The solution is trivial.
 */
template <typename ScalarType, Kokkos::utils::concepts::MemorySpace Mem>
struct FakeFEMLaplacian1D
{
    using matrix_t = KokkosSparse::CrsMatrix<
        /* scalar type */ ScalarType,
        /* index type  */ int,
        /* device      */ Mem
    >;
    using graph_t   = typename matrix_t::staticcrsgraph_type;
    using row_map_t = typename graph_t::row_map_type;
    using entries_t = typename graph_t::entries_type;
    using rhs_t     = Kokkos::View<ScalarType*, Mem>;

    //! We mandate @c Kokkos::complex<double>.
    static_assert(std::same_as<ScalarType, Kokkos::complex<double>>);

    matrix_t mat;
    rhs_t    rhs;
    rhs_t    guess;

    template <Kokkos::utils::concepts::ExecutionSpace Exec>
    static FakeFEMLaplacian1D create(const Exec& exec, const typename Exec::size_type size)
    {
        const auto nnz = 3 * size - 2;

        // NOLINTBEGIN(misc-const-correctness)
        typename row_map_t::non_const_type             row_map(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "row map" ), size + 1);
        typename entries_t::non_const_type             entries(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "entries" ), nnz);
        typename matrix_t::values_type::non_const_type values (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "values"  ), nnz);
        typename rhs_t::non_const_type                 rhs_   (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rhs"     ), size);
        typename rhs_t::non_const_type                 guess_ (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "guess"   ), size);
        // NOLINTEND(misc-const-correctness)

        Kokkos::parallel_for(
            "FakeFEMLaplacian1D",
            Kokkos::RangePolicy(exec, 0, size),
            KOKKOS_LAMBDA(const typename Exec::size_type irow)
            {
                if(irow == 0)
                {
                    row_map(0) = 0;
                    row_map(1) = 2;

                    entries(0) = 0;
                    entries(1) = 1;

                    values(0) = 1.;
                    values(1) = 0.;

                    rhs_(0) = 0.;
                }
                else if(irow < size - 1)
                {
                    const auto offset = 3 * (irow-1) + 2;

                    row_map(irow + 1) = offset + 3;

                    entries(offset + 0) = irow - 1;
                    entries(offset + 1) = irow + 0;
                    entries(offset + 2) = irow + 1;

                    values(offset + 0) = irow == 1 ? 0 : -1.;
                    values(offset + 1) =  2.;
                    values(offset + 2) = -1.;

                    rhs_(irow) = 0.;
                }
                else
                {
                    const auto offset = 3 * (irow-1) + 2;

                    row_map(irow + 1) = offset + 2;

                    entries(offset + 0) = irow - 1;
                    entries(offset + 1) = irow + 0;

                    values(offset + 0) = -1.;
                    values(offset + 1) =  1.;

                    rhs_(irow) = ScalarType{2. / size, 2. / size};
                }

                guess_(irow) = rhs_(irow);
            }
        );

        return {
            .mat   = matrix_t("matrix", size, size, nnz, std::move(values), std::move(row_map), std::move(entries)),
            .rhs   = std::move(rhs_),
            .guess = std::move(guess_)
        };
    }
};

template <Kokkos::utils::concepts::ExecutionSpace Exec>
struct NbyNSolverTestHelper
{
    using execution_space = Exec;
    using memory_space    = typename execution_space::memory_space;
    using initializer_t   = FakeFEMLaplacian1D<Kokkos::complex<double>, memory_space>;
};

template <typename SolverType>
struct NbyNSolverTest
{
    using solver_t = SolverType;

    //! If it's an aggregate, we simply initialize the members.
    template <Kokkos::utils::concepts::ExecutionSpace Exec, typename MatType, typename RhsType> requires std::is_aggregate_v<SolverType>
    static auto get_solver(const Exec&, MatType&& mat, RhsType&& rhs) {
        return solver_t{{}, std::forward<MatType>(mat), std::forward<RhsType>(rhs)};
    }

    template <Kokkos::utils::concepts::ExecutionSpace Exec, typename MatType, typename RhsType>
    static auto get_solver(const Exec& exec, MatType&& mat, RhsType&& rhs) {
        return solver_t{exec, std::forward<MatType>(mat), std::forward<RhsType>(rhs)};
    }

    struct NoOp
    {
        template <typename T>
        void operator()(T&&) const {}
    };

    template <Kokkos::utils::concepts::ExecutionSpace Exec, typename Callback = NoOp>
    static auto run(const Exec& exec, const typename Exec::size_type nrows, const typename SolverType::Parameters& params, Callback&& callback = Callback{})
    {
        auto system = NbyNSolverTestHelper<Exec>::initializer_t::create(exec, nrows);

        auto solver = get_solver(exec, std::move(system.mat), std::move(system.rhs));

        std::forward<Callback>(callback)(solver);

        Kokkos::utils::timer::Timer<void> timer;
        timer.start();
        const auto [res_nrm2, num_iters] = solver.apply(exec, system.guess, params);
        timer.stop();
        const auto elapsed = timer.template duration<Kokkos::utils::timer::seconds>();

        return std::tuple{elapsed, res_nrm2, num_iters, std::move(system.guess)};
    }
};

//! Relative difference between two @c Kokkos::complex.
template <std::floating_point T>
KOKKOS_FORCEINLINE_FUNCTION
auto relDifference(const Kokkos::complex<T>& val1, const Kokkos::complex<T>& val2)
{
    constexpr auto epsilon = Kokkos::Experimental::epsilon_v<T>;
    return abs(val1 - val2) / (epsilon + (abs(val1) <= abs(val2) ? abs(val1) : abs(val2)));
}

//! Helper for writing tests that use @ref tests::cg::NbyNSolverTest. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define RUN_AND_CHECK(_exec_, _nrows_, _tol_, _expt_niters_)                                           \
    const auto [elapsed, res_nrm2, num_iters, sol] = this->run(                                        \
        _exec_, _nrows_,                                                                               \
        {_tol_, 2 * _expt_niters_}                                                                     \
    );                                                                                                 \
                                                                                                       \
    EXPECT_LT(res_nrm2,  _tol_);                                                                       \
    EXPECT_GT(res_nrm2,  0) << "We never expect an exact solution.";                                   \
    EXPECT_EQ(num_iters, _expt_niters_);                                                               \
                                                                                                       \
    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, sol); \
    for(typename std::remove_cvref_t<decltype(_exec_)>::size_type irow = 0; irow < _nrows_; ++irow) {  \
        const Kokkos::complex<double> value {double(2 * irow) / _nrows_, double(2 * irow) / _nrows_};  \
        EXPECT_LE(                                                                                     \
            tests::cg::relDifference(mirror(irow), value),                                             \
            _tol_                                                                                      \
        ) << mirror(irow) << " not close to " << value;                                                \
    }

} // namespace tests::cg

#endif // GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
