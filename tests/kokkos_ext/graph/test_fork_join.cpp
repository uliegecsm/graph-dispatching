#include <vector>

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-result")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/fork_join.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/graph/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/LoadCheckAdd.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c exec::fork_join by @c Kokkos::Experimental::GraphContext
 * ----------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::GraphContext properly customizes
 * @c exec::fork_join.
 *
 * The tests can be found in @ref tests/kokkos_ext/graph/test_fork_join.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext::graph {

using namespace Kokkos::utils::callbacks;

class ForkJoinTest
    : public impl::GraphContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent, ProfileEvent>;
    using variant_t = std::variant<BeginFenceEvent, BeginParallelForEvent, ProfileEvent>;

    using value_t = typename view_s_t::value_type;

    static constexpr bool on_device = ::tests::utils::on_device<execution_space>();
};

//! @test Use @c exec::fork_join with a diamond topology.
TEST_F(ForkJoinTest, diamond) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    auto sndr =
        ::stdexec::schedule(grc.get_scheduler())
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 0, .value = 4, .data = data.data()})
        | ::exec::fork_join(
            ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC,
            ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC)
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 6, .value = 3, .data = data.data()});

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { ::stdexec::sync_wait(std::move(sndr)); }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 9);
}

//! @test Use @c exec::fork_join with a double diamond topology.
TEST_F(ForkJoinTest, double_diamond) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    auto sndr =
        ::stdexec::schedule(grc.get_scheduler())
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 0, .value = 4, .data = data.data()})
        | ::exec::fork_join(
            ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC,
            ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC)
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 6, .value = 3, .data = data.data()})
        | ::exec::fork_join(
            ::stdexec::continues_on(grc.get_scheduler()) | ADD_BULK(3),
            ::stdexec::continues_on(grc.get_scheduler()) | ADD_BULK(3))
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 15, .value = 5, .data = data.data()});

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(execution_space{}));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { ::stdexec::sync_wait(std::move(sndr)); }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 20);
}

} // namespace tests::kokkos_ext::graph
