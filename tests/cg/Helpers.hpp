#ifndef GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP

#include "Kokkos_Core.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

namespace tests::cg
{
/**
 * @brief Create a 2-by-2 system to test solvers.
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
template <typename scalar_t, typename DeviceType>
struct TwoByTwo
{
    using matrix_t = KokkosSparse::CrsMatrix<
        /* scalar type */ scalar_t,
        /* index type  */ int,
        /* device      */ DeviceType
    >;
    using graph_t   = typename matrix_t::staticcrsgraph_type;
    using row_map_t = typename graph_t::row_map_type;
    using entries_t = typename graph_t::entries_type;
    using values_t  = typename matrix_t::values_type;

    values_t rhs;
    matrix_t matrix;
    values_t guess;

    static constexpr scalar_t sol_0 = 1. / 11.;
    static constexpr scalar_t sol_1 = 7. / 11.;

    template <typename Exec>
    TwoByTwo(const Exec& exec)
    {
        typename row_map_t::non_const_type row_map(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "row map" ), 3);
        typename entries_t::non_const_type entries(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "entries" ), 4);
        typename values_t::non_const_type  values (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "value"   ), 4);
        typename values_t::non_const_type  rhs_   (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rhs"     ), 2);
        typename values_t::non_const_type  guess_ (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "guess"   ), 2);

        Kokkos::parallel_for(
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

                rhs_(0) = 1.; rhs_(1) = 2.;

                guess_(0) = 2.; guess_(1) = 1.;
            }
        );

        this->rhs    = std::move(rhs_);
        this->matrix = matrix_t("matrix", 2, 2, 4, std::move(values), std::move(row_map), std::move(entries));
        this->guess  = std::move(guess_);
    }
};

} // namespace tests::cg

#endif // GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
