#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/EventRegionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Functors.hpp"
#define GRAPH_DISPATCHING_ALGORITHMS_CG_CGSINGLEQUEUE_ENABLE_SCOPEDREGION_IN_LOOP
#include "algorithms/cg/SingleQueue.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with a single @c Kokkos execution space instance but avoiding TPLs
 * --------------------------------------------------------------------------------------------
 *
 * Similar to @ref tests/cg/test_single_queue.cpp but we avoid TPLs like @c cuSPARSE.
 *
 * The test can be found in @ref tests/cg/test_single_queue_no_tpl.cpp.
 */

namespace tests::cg
{

using solver_t = algorithms::cg::CGSingleQueue<
    NbyNSolverTestHelper::initializer_t::matrix_t,
    NbyNSolverTestHelper::initializer_t::rhs_t,
    ::algorithms::cg::Spmv,
    ::algorithms::cg::Dot,
    ::algorithms::cg::Axpby
>;

using namespace Kokkos::utils::callbacks;

class CGSingleQueueNoTPLTest : public ::testing::Test,
                               public NbyNSolverTest<solver_t>,
                               public Kokkos::utils::tests::scoped::callbacks::Manager
{
public:
    using event_types_list_t = Kokkos::Impl::type_list<BeginFenceEvent, BeginParallelForEvent, BeginParallelReduceEvent, PushRegionEvent, PopRegionEvent>;

    using event_in_region_recorder_t = RecorderListener<EventRegionMatcher<EventNameMatcher>, event_types_list_t>;
};

TEST_F(CGSingleQueueNoTPLTest, 10x10)
{
    const NbyNSolverTestHelper::execution_space exec {};

    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "CGSingleQueue - iter 7"}};
    const auto recorder = std::make_shared<event_in_region_recorder_t>(std::move(matcher));

    Kokkos::utils::callbacks::Manager::register_listener(recorder);

    RUN_AND_CHECK(exec, 10, 1.e-12, 9)

    Kokkos::utils::callbacks::Manager::unregister_listener(recorder.get());

    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, "algorithms::cg::spmv"),

            MATCHER_FOR_BEGIN_PRED(exec, "algorithms::cg::dot"),

            MATCHER_FOR_BEGIN_FENCE(exec, "waiting for dot (quadratic)"),

            MATCHER_FOR_BEGIN_PFOR(exec, "algorithms::cg::axpby"),
            MATCHER_FOR_BEGIN_PFOR(exec, "algorithms::cg::axpby"),

            MATCHER_FOR_BEGIN_PRED(exec, "algorithms::cg::dot"),

            MATCHER_FOR_BEGIN_FENCE(exec, "waiting for dot (norm)"),

            MATCHER_FOR_BEGIN_PFOR(exec, "algorithms::cg::axpby")
        )
    );
}

} // namespace tests::cg
