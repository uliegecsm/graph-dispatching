#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-utils/callbacks/EventNameMatcher.hpp"
#include "kokkos-utils/callbacks/EventRegionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "algorithms/cg/Graph.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/cg/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Conjugate gradient solver with @c Kokkos::Graph
 * -----------------------------------------------
 *
 * Implement a portable conjugate gradient (CG) solver using @c Kokkos::Graph.
 *
 * The test can be found in @ref tests/cg/test_graph.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::cg
{

using namespace Kokkos::utils::callbacks;

struct UseHostNode   { static constexpr bool value = true; };
struct UseDeviceNode { static constexpr bool value = false; };

using helper_t = ::tests::cg::NbyNSolverTestHelper<execution_space>;

template <typename HostOrDeviceNode>
class CGGraphTest : public ::testing::Test,
                    public NbyNSolverTest<algorithms::cg::CGGraph<typename helper_t::initializer_t::matrix_t, typename helper_t::initializer_t::rhs_t, HostOrDeviceNode::value>>,
                    public Kokkos::utils::tests::scoped::callbacks::Manager,
                    public Kokkos::utils::tests::scoped::ExecutionSpace<execution_space>
{
public:
    using event_types_list_t = Kokkos::Impl::type_list<BeginFenceEvent, BeginParallelForEvent, BeginParallelReduceEvent, PushRegionEvent, PopRegionEvent>;

    using event_in_region_recorder_t = RecorderListener<EventRegionMatcher<EventNameMatcher>, event_types_list_t>;
};

using CGGraphTestTypes = ::testing::Types<UseHostNode, UseDeviceNode>;

TYPED_TEST_SUITE(CGGraphTest, CGGraphTestTypes);

TYPED_TEST(CGGraphTest, 10x10)
{
    EventRegionMatcher matcher{.matcher = EventNameMatcher{.name = "CGGraph - loop"}};

    const auto recorder = std::make_shared<typename TestFixture::event_in_region_recorder_t>(std::move(matcher));

    Manager::register_listener(recorder);

    RUN_AND_CHECK(this->exec, 10000, 1.e-12, 9999)

    Manager::unregister_listener(recorder.get());

    /// Everything happens in the graph, and there is no marker yet. The only event is our fence before computing the
    /// convergence criterion.
    ASSERT_THAT(
        recorder->recorded_events,
        ::testing::Contains(MATCHER_FOR_BEGIN_FENCE(this->exec, "fencing before evaluating convergence")).Times(9999)
    );
}

} // namespace tests::cg
