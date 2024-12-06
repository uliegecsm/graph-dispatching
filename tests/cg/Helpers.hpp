#ifndef GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP

#include "Kokkos_Core.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

namespace tests::cg
{
/**
 * @brief Create a N-by-N tri-diagonal system to test solvers.
 *
 * @todo Say that it's from a 1D Laplacian.
 * @todo Change class name.
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

    template <typename Exec>
    TwoByTwo(const Exec& exec, const size_t size)
    {
        typename row_map_t::non_const_type row_map(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "row map" ), size + 1);
        typename entries_t::non_const_type entries(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "entries" ), 3 * size - 2);
        typename values_t::non_const_type  values (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "value"   ), 3 * size - 2);
        typename values_t::non_const_type  rhs_   (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rhs"     ), size);
        typename values_t::non_const_type  guess_ (Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "guess"   ), size);

        Kokkos::parallel_for(
            Kokkos::RangePolicy(exec, 0, size),
            KOKKOS_LAMBDA(const typename Exec::size_type irow)
            {
                if(irow == 0)
                {
                    row_map(0) = 0;
                    row_map(1) = 2;

                    entries(0) = 0;
                    entries(1) = 1;

                    values(0) =  1.;
                    values(1) = -1.;

                    rhs_(0) = 2.;

                    guess_(0) = 0.1;
                }
                else if(irow < size - 1)
                {
                    const auto offset = 3 * (irow-1) + 2;

                    row_map(irow + 1) = offset + 3;

                    entries(offset + 0) = irow - 1;
                    entries(offset + 1) = irow + 0;
                    entries(offset + 2) = irow + 1;

                    values(offset + 0) = -1.;
                    values(offset + 1) =  2.;
                    values(offset + 2) = -1.;

                    rhs_(irow) = 0.;

                    guess_(irow) = 0.;
                }
                else
                {
                    const auto offset = 3 * (irow-1) + 2;

                    row_map(irow + 1) = offset + 2;

                    entries(offset + 0) = irow - 1;
                    entries(offset + 1) = irow + 0;

                    values(offset + 0) = -1.;
                    values(offset + 1) =  2.;

                    rhs_(irow) = 0.;

                    guess_(irow) = 0.;
                }
            }
        );

        this->rhs    = std::move(rhs_);
        this->matrix = matrix_t("matrix", size, size, entries.size(), std::move(values), std::move(row_map), std::move(entries));
        this->guess  = std::move(guess_);
    }
};

/**
 * @brief Conjugate gradient solver.
 *
 * References:
 *  - https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
 */
template <typename VectorType, typename MatrixType>
struct ConjugateGradientSolverBase
{
    //! Result of the dot product.
    using dot_t    = typename Kokkos::Details::InnerProductSpaceTraits<typename VectorType::non_const_value_type>::dot_type;

    //! Type used to store scalar results needed only on host.
    using device_t = Kokkos::View<dot_t, typename VectorType::memory_space>;

    /// Type used to store scalar results needed on both host and device.
    /// Using a pinned allocation avoids costly memory copies.
    using pinned_t = Kokkos::View<dot_t, Kokkos::SharedHostPinnedSpace>;
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

struct NbyNSolverTestHelper
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;
    using initializer_t   = TwoByTwo<double, memory_space>;
};

template <typename SolverType>
class NbyNSolverTest : public ::testing::Test
{
public:
    static constexpr double tolerance = 1.e-12;

public:
    void run(const size_t nrows)
    {
        const NbyNSolverTestHelper::execution_space exec {};

        NbyNSolverTestHelper::initializer_t system(exec, nrows);

        SolverType solver{.rhs = std::move(system.rhs), .mat = std::move(system.matrix)};

        Kokkos::Timer timer;
        const auto [res_nrm2, num_iters] = solver.apply(exec, system.guess, tolerance, 2 * nrows);
        const auto elapsed = timer.seconds();

        std::cout << "> Convergence in " << elapsed << " seconds, in " << num_iters << " iterations, residual L2 norm is " << res_nrm2 << '.' << std::endl;

        ASSERT_LT(res_nrm2,  tolerance);
        ASSERT_EQ(num_iters, nrows);

        const auto sol_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), system.guess);

        std::cout << "> Solution: [";
        for(size_t irow = 0; irow < nrows; ++irow) std::cout << sol_h(irow) << ", ";
        std::cout << "]" << std::endl;
    }
};

} // namespace tests::cg

#endif // GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
