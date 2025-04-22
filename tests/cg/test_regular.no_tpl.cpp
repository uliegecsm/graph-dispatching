#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/EventRegionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/cg/Helpers.hpp"
#include "tests/cg/Regular.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with regular @c Kokkos execution space instances but avoiding TPLs
 * --------------------------------------------------------------------------------------------
 *
 * Similar to @ref tests/cg/test_regular.cpp but we avoid TPLs like @c cuSPARSE.
 *
 * The test can be found in @ref tests/cg/test_regular.no_tpl.cpp.
 */

namespace tests::cg
{

DEFINE_FUNCTOR(TestsCgSpmv,  ::tests::cg::spmv)
DEFINE_FUNCTOR(TestsCgDot,   ::tests::cg::dot)
DEFINE_FUNCTOR(TestsCgAxpby, ::tests::cg::axpby)

using solver_t = CGRegular<
    NbyNSolverTestHelper::initializer_t::matrix_t,
    NbyNSolverTestHelper::initializer_t::rhs_t,
    TestsCgSpmv,
    TestsCgDot,
    TestsCgAxpby
>;

using namespace Kokkos::utils::callbacks;

class CGRegularNoTPLTest : public NbyNSolverTest<solver_t>,
                           public Kokkos::utils::callbacks::ManagerTestFixture
{
public:
    using event_types_list_t = Kokkos::Impl::type_list<BeginFenceEvent, BeginParallelForEvent, BeginParallelReduceEvent, PushRegionEvent, PopRegionEvent>;

    using event_in_region_recorder_t = RecorderListener<EventRegionMatcher<EventNameMatcher>, event_types_list_t>;
};

TEST_F(CGRegularNoTPLTest, 10x10)
{
    const NbyNSolverTestHelper::execution_space exec {};

#if defined(KOKKOS_ENABLE_CUDA)
    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "CGRegular - iter 7"}};
    const auto recorder = std::make_shared<event_in_region_recorder_t>(std::move(matcher));

    Manager::register_listener(recorder);
#endif

    RUN_AND_CHECK(exec, 10, 1.e-12, 9)

#if defined(KOKKOS_ENABLE_CUDA)
    Manager::unregister_listener(recorder.get());

    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, "tests::cg::spmv"),

            MATCHER_FOR_BEGIN_PRED(exec, "tests::cg::dot"),

            MATCHER_FOR_BEGIN_FENCE(exec, "waiting for dot (quadratic)"),

            MATCHER_FOR_BEGIN_PFOR(exec, "tests::cg::axpby"),
            MATCHER_FOR_BEGIN_PFOR(exec, "tests::cg::axpby"),

            MATCHER_FOR_BEGIN_PRED(exec, "tests::cg::dot"),

            MATCHER_FOR_BEGIN_FENCE(exec, "waiting for dot (norm)"),

            MATCHER_FOR_BEGIN_PFOR(exec, "tests::cg::axpby")
        )
    );
#endif
}

} // namespace tests::cg
