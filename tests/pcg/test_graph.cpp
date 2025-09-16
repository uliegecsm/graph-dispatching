#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/EventRegionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/concepts/View.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Functors.hpp"
#include "algorithms/pcg/Graph.hpp"
#include "algorithms/pcg/Preconditioners.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Preconditioned conjugate gradient solver with @c Kokkos::Graph
 * --------------------------------------------------------------
 *
 * This group of tests check the behavior of @ref algorithms::pcg::PCGSingleQueue.
 *
 * The test can be found in @ref tests/pcg/test_graph.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::pcg
{

using namespace Kokkos::utils::callbacks;

//! Base class for the test setup that does not dependent on the preconditioner type.
class PCGGraphTestNoPrecBase : public ::testing::Test,
                               public Kokkos::utils::tests::scoped::callbacks::Manager,
                               public Kokkos::utils::tests::scoped::ExecutionSpace<execution_space>
{
public:
    using helper_t = cg::NbyNSolverTestHelper<execution_space>;

    using scalar_t = helper_t::initializer_t::rhs_t::non_const_value_type;

    using event_types_list_t = Kokkos::Impl::type_list<BeginFenceEvent, BeginParallelForEvent, BeginParallelReduceEvent, BeginDeepCopyEvent,  PushRegionEvent, PopRegionEvent>;

    using event_in_region_recorder_t = RecorderListener<EventRegionMatcher<EventNameMatcher>, event_types_list_t>;

    using matrix_t = helper_t::initializer_t::matrix_t;
    using rhs_t    = helper_t::initializer_t::rhs_t;
};

//! Base class for the test setup that depends on the preconditioner type.
template <typename Preconditioner>
class PCGGraphTestBase : public PCGGraphTestNoPrecBase
{
public:
    using preconditioner_t = Preconditioner;

    using solver_t = ::algorithms::pcg::PCGGraph<
        matrix_t,
        rhs_t,
        Kokkos::Experimental::Graph<execution_space>,
        preconditioner_t
    >;
};

template <typename Preconditioner>
class PCGGraphTest : public PCGGraphTestBase<Preconditioner>,
                     public cg::NbyNSolverTest<typename PCGGraphTestBase<Preconditioner>::solver_t>
{
protected:
#define GET_EXPECTED_NUM_ITERS(_prec_, values)                                                               \
    static std::unordered_map<size_t, size_t> get_expected_num_iters() requires                              \
        std::same_as<Preconditioner, ::algorithms::pcg::_prec_<typename PCGGraphTestNoPrecBase::matrix_t>> { \
        return {KOKKOS_IMPL_STRIP_PARENS(values)};                                                           \
    }

    GET_EXPECTED_NUM_ITERS(IdentityPreconditioner, ({10, 9}, {90, 89}))
    GET_EXPECTED_NUM_ITERS(DiagonalPreconditioner, ({10, 8}, {90, 88}))
    GET_EXPECTED_NUM_ITERS(JacobiPreconditioner,   ({10, 4}, {90, 39}))

#undef GET_EXPECTED_NUM_ITERS
};

using PCGGraphTestTypes = ::testing::Types<
    ::algorithms::pcg::IdentityPreconditioner<typename PCGGraphTestNoPrecBase::matrix_t>,
    ::algorithms::pcg::DiagonalPreconditioner<typename PCGGraphTestNoPrecBase::matrix_t>,
    ::algorithms::pcg::JacobiPreconditioner  <typename PCGGraphTestNoPrecBase::matrix_t>
>;

TYPED_TEST_SUITE(PCGGraphTest, PCGGraphTestTypes); 

//! @test For a 10-by-10 system.
TYPED_TEST(PCGGraphTest, 10x10)
{
    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "PCGGraph - loop"}};

    const auto recorder = std::make_shared<typename TestFixture::event_in_region_recorder_t>(std::move(matcher));

    Manager::register_listener(recorder);

    RUN_AND_CHECK(this->exec, 10, 1.e-12, TestFixture::get_expected_num_iters()[10])

    Manager::unregister_listener(recorder.get());

    /// Everything happens in the graph, and there is no marker yet. The only event is our fence before computing the
    /// convergence criterion.
    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::Contains(MATCHER_FOR_BEGIN_FENCE(this->exec, "fencing before evaluating convergence")).Times(TestFixture::get_expected_num_iters()[10])
    );
}

//! @test For a 90-by-90 system.
TYPED_TEST(PCGGraphTest, 90x90)
{
    RUN_AND_CHECK(this->exec, 90, 1.e-12, TestFixture::get_expected_num_iters()[90])
}

} // namespace tests::pcg
