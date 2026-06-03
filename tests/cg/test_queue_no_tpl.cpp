#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/EventRegionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Functors.hpp"
#define GRAPH_DISPATCHING_ALGORITHMS_CG_CGQUEUE_ENABLE_SCOPEDREGION_IN_LOOP
#include "algorithms/cg/Queue.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with @c Kokkos execution space instances but avoiding TPLs
 * ------------------------------------------------------------------------------------
 *
 * Similar to @ref tests/cg/test_queue.cpp but we avoid TPLs like @c cuSPARSE.
 *
 * The test can be found in @ref tests/cg/test_queue_no_tpl.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::cg
{

using helper_t = NbyNSolverTestHelper<execution_space>;

using solver_t = algorithms::cg::CGQueue<
    typename helper_t::initializer_t::matrix_t,
    typename helper_t::initializer_t::rhs_t,
    ::algorithms::cg::Spmv,
    ::algorithms::cg::Dot,
    ::algorithms::cg::Axpby
>;

using namespace Kokkos::utils::callbacks;

class CGQueueNoTPLTest : public ::testing::Test,
                               public NbyNSolverTest<solver_t>,
                               public Kokkos::utils::tests::scoped::callbacks::Manager,
                               public utils::ExecutionSpacePoolFixture<execution_space, 2>
{
public:
    using event_types_list_t = Kokkos::Impl::type_list<BeginFenceEvent, BeginParallelForEvent, BeginParallelReduceEvent, PushRegionEvent, PopRegionEvent>;

    using event_in_region_recorder_t = RecorderListener<EventRegionMatcher<EventNameMatcher>, event_types_list_t>;
};

TEST_F(CGQueueNoTPLTest, 10x10)
{
    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "CGQueue - iter 7"}};
    const auto recorder = std::make_shared<event_in_region_recorder_t>(std::move(matcher));

    Kokkos::utils::callbacks::Manager::register_listener(recorder);

    RUN_AND_CHECK(this->pool, 10, 1.e-12, 9)

    Kokkos::utils::callbacks::Manager::unregister_listener(recorder.get());

    const auto& exec_A = pool.get(0);
    const auto& exec_B = pool.get(1);

    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec_A, "algorithms::cg::spmv"),

            MATCHER_FOR_BEGIN_PRED(exec_A, "algorithms::cg::dot"),

            MATCHER_FOR_BEGIN_FENCE(exec_A, "waiting for dot (quadratic)"),

            MATCHER_FOR_BEGIN_PFOR(exec_A, "algorithms::cg::axpby"),
            MATCHER_FOR_BEGIN_PFOR(exec_B, "algorithms::cg::axpby"),

            MATCHER_FOR_BEGIN_PRED(exec_B, "algorithms::cg::dot"),

            MATCHER_FOR_BEGIN_FENCE(exec_B, "waiting for dot (norm)"),

            MATCHER_FOR_BEGIN_PFOR(exec_A, "algorithms::cg::axpby")
        )
    );
}

} // namespace tests::cg
