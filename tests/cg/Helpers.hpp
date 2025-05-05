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

} // namespace tests::cg

#endif // GRAPH_DISPATCHING_TESTS_CG_HELPERS_HPP
