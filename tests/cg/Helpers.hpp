#ifndef GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP

#include "Kokkos_Core.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

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

/**
 * @brief Scalar division on device plus negation, graph-compatible.
 *
 * References:
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu#L103C1-L108C2
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu#L96
 */
template <typename Exec, typename T, typename U, typename V>
decltype(auto) scalar_div_and_neg(Exec&& exec, const T& out, const T& out_neg, const U& x, const V& y)
{
    using execution_space = typename std::remove_cvref_t<Exec>::execution_space;

    return std::forward<Exec>(exec) | Kokkos::Experimental::graph::parallel_for(
        "scalar division + negate",
        Kokkos::RangePolicy<execution_space>(0, 1),
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
decltype(auto) scalar_div(Exec&& exec, const T& out, const U& x, const V& y)
{
    return std::forward<Exec>(exec) | Kokkos::Experimental::graph::parallel_for(
        "scalar division",
        Kokkos::RangePolicy(0, 1),
        KOKKOS_LAMBDA(const int){ out() = x() / y(); }
    );
}

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
    > functor(alpha, mat, vec_x, beta, vec_y, /* rows_per_team */ mat.numRows());

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

template <typename T> requires (!Kokkos::is_view_v<std::remove_cvref_t<T>>)
constexpr decltype(auto) get_value(T&& value) {
    return std::forward<T>(value);
}

template <typename T> requires (Kokkos::is_view_v<std::remove_cvref_t<T>> && std::remove_cvref_t<T>::rank() == 0)
constexpr decltype(auto) get_value(T&& value) {
    return std::forward<T>(value)();
}

//! Equivalent to @c KokkosBlas::axpby, graph-compatible.
template <typename Exec, typename Alpha, typename ViewX, typename Beta, typename ViewY>
decltype(auto) axpby(Exec&& exec, const Alpha& alpha, const ViewX& vec_x, const Beta& beta, const ViewY& vec_y)
{
    using execution_space = typename std::remove_cvref_t<Exec>::execution_space;

    return std::forward<Exec>(exec) | Kokkos::Experimental::graph::parallel_for(
        "AXPBY",
        Kokkos::RangePolicy<execution_space>(0, vec_x.size()),
        KOKKOS_LAMBDA(const typename execution_space::size_type index) {
            vec_y(index) = get_value(alpha) * vec_x(index) + get_value(beta) * vec_y(index);
        }
    );
}

} // namespace tests::cg

#endif // GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
