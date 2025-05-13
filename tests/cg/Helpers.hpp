#ifndef GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP

#include "Kokkos_InnerProductSpaceTraits.hpp"

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"

#include "kokkos-utils/concepts/ExecutionSpace.hpp"
#include "kokkos-utils/concepts/MemorySpace.hpp"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

namespace tests::cg
{
/**
 * @brief Create a N-by-N tri-diagonal symmetric system to test solvers.
 *
 * Inspired by a 1D Laplacian FEM problem with a Dirichlet BC on the left and
 * a non-homogenous Neumann on the right.
 *
 * @verbatim
 * 1  0                     0
 * 0  2 -1                  0
 *       *                  *
 *           *              *
 *              *           *
 *             -1  2 -1     0
 *                -1  1     2 + 2i
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

                    rhs_(irow) = ScalarType{2., 2.};
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

/**
 * @brief Conjugate gradient solver base.
 *
 * References:
 *  - https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  - https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
 */
template <typename MatrixType, typename VectorType> requires std::same_as<typename MatrixType::memory_space, typename VectorType::memory_space>
struct ConjugateGradientSolverBase
{
    //! Result of @c nrm2.
    using mag_t = typename Kokkos::Details::InnerProductSpaceTraits<typename VectorType::non_const_value_type>::mag_type;

    //! Result of @c dot.
    using dot_t = typename Kokkos::Details::InnerProductSpaceTraits<typename VectorType::non_const_value_type>::dot_type;
};

struct NbyNSolverTestHelper
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;
    using initializer_t   = FakeFEMLaplacian1D<Kokkos::complex<double>, memory_space>;
};

template <typename SolverType>
class NbyNSolverTest : public virtual ::testing::Test
{
public:
    static constexpr SolverType::mag_t tolerance = 1.e-12;

public:
    template <Kokkos::utils::concepts::ExecutionSpace Exec>
    auto run(const Exec& exec, const typename Exec::size_type nrows)
    {
        auto system = NbyNSolverTestHelper::initializer_t::create(exec, nrows);

        static_assert(std::is_aggregate_v<SolverType>);

        SolverType solver{{}, std::move(system.mat), std::move(system.rhs)};

        const Kokkos::Timer timer;
        const auto [res_nrm2, num_iters] = solver.apply(exec, system.guess, tolerance, 2 * nrows);
        const auto elapsed = timer.seconds();

        EXPECT_LT(res_nrm2,  tolerance);
        EXPECT_EQ(num_iters, nrows - 1);

        //! Check that the solution is correct. @todo Make it more efficient.
        const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, system.guess);
        for(typename Exec::size_type irow = 0; irow < nrows; ++irow)
        {
            EXPECT_NEAR(mirror(irow).real(), 2 * irow, 1.e-14);
            EXPECT_NEAR(mirror(irow).imag(), 2 * irow, 1.e-14);
        }

        return elapsed;
    }
};

/**
 * @brief Perform a sparse matrix-vector multiply, graph-compatible.
 *
 * The implementation relies on the @c KokkosSparse kernel, inspired by:
 *  - https://github.com/trilinos/Trilinos/blob/62023ad68e09a2972240971c40be34465010d6f3/packages/kokkos-kernels/perf_test/sparse/KokkosSparse_spmv_struct_tuning.cpp#L191.
 *
 * @note Therefore, this version will never rely on a TPL. This is because @c KokkosKernels is not yet graph-compatible;
 *       calling @c KokkosSparse::spmv using a capture stream might not work either - but could be tried.
 *
 * @warning Not tuned to be efficient.
 */
template <typename Pred, typename Handle, typename Alpha, typename AMatrix, typename XVector, typename Beta, typename YVector>
decltype(auto) spmv(const Pred& pred, const Handle&, const char mode[], Alpha&& alpha, AMatrix&& mat, XVector&& vec_x, Beta&& beta, YVector&& vec_y)
{
    using execution_space = std::conditional_t<
        Kokkos::utils::concepts::ExecutionSpace<Pred>,
        Pred,
        typename Pred::execution_space
    >;

    if(mode[0] != 'N') Kokkos::abort("Unsupported mode.");

    const auto rows_per_team = mat.numRows();

    KokkosSparse::Impl::SPMV_Functor< // NOLINT(misc-const-correctness)
        execution_space,
        std::remove_cvref_t<AMatrix>,
        std::remove_cvref_t<XVector>,
        std::remove_cvref_t<YVector>,
        1     /* dobeta */,
        false /* conjugate */
    > functor(
        std::forward<Alpha>(alpha),
        std::forward<AMatrix>(mat),
        std::forward<XVector>(vec_x),
        std::forward<Beta>(beta),
        std::forward<YVector>(vec_y),
        rows_per_team
    );

    if constexpr (Kokkos::utils::concepts::ExecutionSpace<Pred>) {
        Kokkos::parallel_for(
            "tests::cg::spmv",
            Kokkos::TeamPolicy<execution_space>(pred, 1, Kokkos::AUTO),
            std::move(functor)
        );
    } else {
        return pred.then_parallel_for(
            "tests::cg::spmv",
            Kokkos::TeamPolicy<execution_space>(1, Kokkos::AUTO),
            std::move(functor)
        );
    }
}

//! Dot product, graph-compatible.
template <typename Pred, typename Result, typename ViewX, typename ViewY>
decltype(auto) dot(const Pred& pred, Result&& result, ViewX&& vec_x, ViewY&& vec_y)
{
    using execution_space = std::conditional_t<
        Kokkos::utils::concepts::ExecutionSpace<Pred>,
        Pred,
        typename Pred::execution_space
    >;

    KokkosBlas::Impl::DotFunctor< // NOLINT(misc-const-correctness)
        std::remove_cvref_t<Result>,
        std::remove_cvref_t<ViewX>,
        std::remove_cvref_t<ViewY>,
        typename execution_space::size_type
    > functor(std::forward<ViewX>(vec_x), std::forward<ViewX>(vec_y));

    if constexpr (Kokkos::utils::concepts::ExecutionSpace<Pred>) {
        Kokkos::parallel_reduce(
            "tests::cg::dot",
            Kokkos::RangePolicy<execution_space>(pred, 0, functor.m_x.size()),
            std::move(functor),
            std::forward<Result>(result)
        );
    } else {
        return pred.then_parallel_reduce(
            "tests::cg::dot",
            Kokkos::RangePolicy<execution_space>(0, functor.m_x.size()),
            std::move(functor),
            std::forward<Result>(result)
        );
    }
}

template <typename T> requires (!Kokkos::is_view_v<std::remove_cvref_t<T>>)
constexpr decltype(auto) get_value(T&& value) {
    return std::forward<T>(value);
}

template <typename T> requires (Kokkos::is_view_v<std::remove_cvref_t<T>> && std::remove_cvref_t<T>::rank() == 0)
constexpr decltype(auto) get_value(T&& value) {
    return std::forward<T>(value)();
}

template <typename Alpha, typename ViewX, typename Beta, typename ViewY>
struct Axpby
{
    Alpha alpha;
    ViewX vec_x;
    Beta  beta;
    ViewY vec_y;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        vec_y(index) = get_value(alpha) * vec_x(index) + get_value(beta) * vec_y(index);
    }
};

/**
 * @brief Equivalent to @c KokkosBlas::axpby, graph-compatible.
 *
 * @note @c alpha and @c beta can be rank-0 views (for coefficients that vary accross multiple re-submissions).
 */
template <typename Pred, typename Alpha, typename ViewX, typename Beta, typename ViewY>
decltype(auto) axpby(const Pred& pred, Alpha&& alpha, ViewX&& vec_x, Beta&& beta, ViewY&& vec_y)
{
    using execution_space = std::conditional_t<
        Kokkos::utils::concepts::ExecutionSpace<Pred>,
        Pred,
        typename Pred::execution_space
    >;

    const auto size = vec_x.size();

    Axpby functor{ // NOLINT(misc-const-correctness)
        .alpha = std::forward<Alpha>(alpha),
        .vec_x = std::forward<ViewX>(vec_x),
        .beta  = std::forward<Beta>(beta),
        .vec_y = std::forward<ViewY>(vec_y)
    };

    if constexpr (Kokkos::utils::concepts::ExecutionSpace<Pred>) {
        Kokkos::parallel_for(
            "tests::cg::axpby",
            Kokkos::RangePolicy<execution_space>(pred, 0, size),
            std::move(functor)
        );
    } else {
        return pred.then_parallel_for(
            "tests::cg::axpby",
            Kokkos::RangePolicy<execution_space>(0, size),
            std::move(functor)
        );
    }
}

//! Helper to defined a new type whose call operator wraps another callable.
#define DEFINE_FUNCTOR(_for_, _with_)                     \
    struct _for_                                          \
    {                                                     \
        template <typename... Args>                       \
        decltype(auto) operator()(Args&&... args) const { \
            return _with_(std::forward<Args>(args)...);   \
        }                                                 \
    };

} // namespace tests::cg

#endif // GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
