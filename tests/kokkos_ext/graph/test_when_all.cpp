#include <vector>

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
 * Customization of @c when_all by @c Kokkos::Experimental::GraphContext
 * ---------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::GraphContext properly customizes
 * @c when_all.
 *
 * The tests can be found in @ref tests/kokkos_ext/graph/test_when_all.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class WhenAllTest
    : public impl::GraphContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<execution_space>,
        BeginFenceEvent, BeginParallelForEvent, ProfileEvent
    >;
    using variant_t = std::variant<BeginFenceEvent, BeginParallelForEvent, ProfileEvent>;

    static constexpr bool on_device = ::tests::utils::on_device<execution_space>();
};

//! Check basic traits of a @c when_all sender.
template <::stdexec::sender T>
constexpr bool check_traits(const T &) {
    static_assert(std::same_as<
                  decltype(::stdexec::get_completion_domain<::stdexec::set_value_t>(
                      ::stdexec::get_env(std::declval<T>()), ::stdexec::env<>{})),
                  Kokkos::Experimental::details::graph::Domain
    >);

    static_assert(!tests::stdexec::has_completion_scheduler_for<T, ::stdexec::set_value_t>);

    return true;
}

//! @test Check that @ref Kokkos::Experimental::GraphContext does its duty well when used with a single-branch @c when_all.
TEST_F(WhenAllTest, one_branch) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    auto when_all = ::stdexec::when_all(::stdexec::schedule(grc.get_scheduler()) | ADD_THEN);

    static_assert(check_traits(when_all));

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::utils::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "when_all")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record(
            [when_all = std::move(when_all)]() mutable { ::stdexec::sync_wait(std::move(when_all)); }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 1);
}

//! @test Check that @ref Kokkos::Experimental::GraphContext does its duty well when used with a two-branches @c when_all.
TEST_F(WhenAllTest, two_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    auto branch_a = ::stdexec::schedule(grc.get_scheduler()) | ADD_THEN_ATOMIC;
    auto branch_b = ::stdexec::schedule(grc.get_scheduler()) | ADD_THEN_ATOMIC;
    auto when_all = ::stdexec::when_all(std::move(branch_a), std::move(branch_b));

    static_assert(check_traits(when_all));

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::utils::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "when_all")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record(
            [when_all = std::move(when_all)]() mutable { ::stdexec::sync_wait(std::move(when_all)); }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 2);
}

//! @test Check that @ref Kokkos::Experimental::GraphContext does its duty well when used with a three-branches @c when_all.
TEST_F(WhenAllTest, three_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    auto branch_a = ::stdexec::schedule(grc.get_scheduler()) | ADD_THEN_ATOMIC;
    auto branch_b = ::stdexec::schedule(grc.get_scheduler()) | ADD_THEN_ATOMIC;
    auto branch_c = ::stdexec::schedule(grc.get_scheduler()) | ADD_THEN_ATOMIC;
    auto when_all = ::stdexec::when_all(std::move(branch_a), std::move(branch_b), std::move(branch_c));

    static_assert(check_traits(when_all));

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::utils::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "when_all")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record(
            [when_all = std::move(when_all)]() mutable { ::stdexec::sync_wait(std::move(when_all)); }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 3);
}

//! @test The topology is a single node in the single @c when_all branch, and it is followed by a single sender after the @c when_all on the inline scheduler.
TEST_F(WhenAllTest, one_branch_and_one_after_on_inline_scheduler) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    auto when_all =
        ::stdexec::when_all(::stdexec::schedule(grc.get_scheduler()) | ADD_THEN)
        | ::stdexec::continues_on(::stdexec::inline_scheduler{})
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<value_t, false>{.prev = 1, .value = 4, .data = data.data()});

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::utils::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "when_all")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record(
            [when_all = std::move(when_all)]() mutable { ::stdexec::sync_wait(std::move(when_all)); }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 5);
}

/**
 * @test The join topology is a single node in each of the two @c when_all branches, followed by a single node after the @c when_all.
 *
 * @verbatim
 * A   B
 *  \ /
 *   C
 * @endverbatim
 */
TEST_F(WhenAllTest, join_topology) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    using functor_t = ::tests::utils::LoadCheckAddFunctor<value_t, on_device>;

    auto when_all = ::stdexec::when_all(
                        ::stdexec::schedule(grc.get_scheduler()) | ADD_THEN_ATOMIC,
                        ::stdexec::schedule(grc.get_scheduler()) | ADD_THEN_ATOMIC)
                  | ::stdexec::continues_on(grc.get_scheduler())
                  | ::stdexec::then(functor_t{.prev = 2, .value = 4, .data = data.data()});

    using outer_0 = ::stdexec::connect_result_t<decltype(when_all), ::tests::stdexec::SinkReceiver>;
    static_assert(::stdexec::__is_instance_of<outer_0, Kokkos::Experimental::details::graph::ThenOpState>);

    static_assert(impl::check_node_type<
                  outer_0,
                  const utils::then_node_t<execution_space, functor_t, utils::aggregate_node_t<execution_space>> &
    >());

    static_assert(std::same_as<
                  ::stdexec::__demangle_t<outer_0>,
                  Kokkos::Experimental::details::graph::ThenOpState<
                      Kokkos::Experimental::details::graph::Scheduler<execution_space>,
                      ::tests::stdexec::basic_sender<
                          ::stdexec::continues_on_t,
                          Kokkos::Experimental::details::graph::Scheduler<execution_space>,
                          ::tests::stdexec::basic_sender<
                              ::stdexec::schedule_from_t,
                              ::stdexec::__,
                              ::tests::stdexec::basic_sender<
                                  ::stdexec::when_all_t,
                                  ::stdexec::__,
                                  ::tests::stdexec::basic_sender<
                                      ::stdexec::then_t,
                                      ::tests::ThenFunctor<atomic<view_s_t>>,
                                      Kokkos::Experimental::details::graph::Scheduler<execution_space>::Sender
                                  >,
                                  ::tests::stdexec::basic_sender<
                                      ::stdexec::then_t,
                                      ::tests::ThenFunctor<atomic<view_s_t>>,
                                      Kokkos::Experimental::details::graph::Scheduler<execution_space>::Sender
                                  >
                              >
                          >
                      >,
                      ::tests::stdexec::SinkReceiver,
                      ::tests::utils::LoadCheckAddFunctor<int, on_device>
                  >
    >);

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::utils::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record(
            [when_all = std::move(when_all)]() mutable { ::stdexec::sync_wait(std::move(when_all)); }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 6);
}

} // namespace tests::kokkos_ext
