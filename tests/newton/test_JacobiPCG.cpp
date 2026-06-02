#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "KokkosSparse_CrsMatrix.hpp"

#include "plog/Formatters/TxtFormatter.h"
#include "plog/Initializers/ConsoleInitializer.h"
#include "plog/Log.h"

#include "algorithms/cg/Functors.hpp"
#include "algorithms/cg/Queue.hpp"
#include "algorithms/newton/Solver.hpp"
#include "algorithms/pcg/Graph.hpp"
#include "algorithms/pcg/Preconditioners.hpp"
#include "algorithms/pcg/Queue.hpp"
#include "apps/heat/NonLinear1DHeatTransfer.hpp"

#include "tests/newton/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Newton solver using a Preconditioned Conjugate Gradient (PCG) with Jacobi as preconditioner
 * -------------------------------------------------------------------------------------------
 *
 * The tests can be found in @ref tests/newton/test_JacobiPCG.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space = typename execution_space::memory_space;

namespace tests::newton {

template <typename UseGraph>
struct NewtonPCGTest
    : public ::testing::Test
    , public utils::ExecutionSpacePoolFixture<execution_space, 2> {
   public:
    using problem_t = ::apps::heat::NonLinear1DHeatTransfer<memory_space, execution_space, UseGraph::value>;

    using preconditioner_t = ::algorithms::pcg::JacobiPreconditioner<typename problem_t::local_matrix_t>;

    using solver_graph_t = algorithms::pcg::PCGGraph<
        typename problem_t::local_matrix_t,
        typename problem_t::scalar_1d_view_t,
        Kokkos::Experimental::Graph<execution_space>,
        preconditioner_t
    >;

    using solver_queue_t = algorithms::pcg::PCGQueue<
        typename problem_t::local_matrix_t,
        typename problem_t::scalar_1d_view_t,
        preconditioner_t,
        ::algorithms::cg::Spmv,
        ::algorithms::cg::Dot,
        ::algorithms::cg::Axpby
    >;

    using linear_solver_t = std::conditional_t<UseGraph::value, solver_graph_t, solver_queue_t>;

    using subtract_t = Subtract<typename problem_t::scalar_1d_view_t, typename problem_t::scalar_1d_view_t>;

    void SetUp() override {
        logger = std::make_unique<::plog::Logger<PLOG_DEFAULT_INSTANCE_ID>>(::plog::debug);
        appender = std::make_unique<::plog::ConsoleAppender<::plog::TxtFormatter>>(::plog::streamStdOut);
        logger->addAppender(appender.get());
    }

    void TearDown() override {
        logger.reset();
        appender.reset();
    }

    std::unique_ptr<::plog::Logger<PLOG_DEFAULT_INSTANCE_ID>> logger = nullptr;
    std::unique_ptr<::plog::ConsoleAppender<::plog::TxtFormatter>> appender = nullptr;
};

using NewtonPCGTestTypes = ::testing::Types<std::integral_constant<bool, true>, std::integral_constant<bool, false>>;

TYPED_TEST_SUITE(NewtonPCGTest, NewtonPCGTestTypes);

//! @test Check that @ref ::algorithms::newton::Solver works with @ref algorithms::pcg::JacobiPreconditioner.
TYPED_TEST(NewtonPCGTest, Jacobi) {
    static_assert(Kokkos::Impl::is_specialization_of_v<
                  typename TestFixture::preconditioner_t,
                  ::algorithms::pcg::JacobiPreconditioner
    >);

    using newton_t = ::algorithms::newton::Solver<
        typename TestFixture::problem_t,
        typename TestFixture::linear_solver_t,
        typename TestFixture::subtract_t
    >;

    const auto& exec_A = this->pool.get(0);

    typename TestFixture::problem_t problem{exec_A, typename TestFixture::problem_t::Parameters{.num_elems = 100}};
    typename TestFixture::linear_solver_t linear_solver{exec_A, problem.local_matrix, problem.local_rhs};

    const newton_t solver{.problem = std::move(problem), .linear_solver = std::move(linear_solver)};

    const auto [res_nrm2, num_iters] =
        solver.solve(this->pool, {.tolerance = 1.e-6, .max_iters = 10}, {.tolerance = 1.e-6, .max_iters = 100});

    const auto sol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, solver.problem.local_sol);

    ASSERT_EQ(num_iters, 3);
    ASSERT_EQ(sol(0), 1.);

    for (size_t ielem = 0; ielem < sol.size() - 1; ++ielem) {
        ASSERT_GT(sol(ielem), sol(ielem + 1)) << "The solution should be decaying.";
    }
}

} // namespace tests::newton
