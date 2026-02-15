#include <vector>

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-result")
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

    static constexpr bool on_device = ::tests::utils::on_device<execution_space>();
};

//! @test A @c exec::fork_join with 3 branches and no sender before or after the fork.
TEST_F(ForkJoinTest, three_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    auto sndr = ::stdexec::schedule(grc.get_scheduler())
              | ::exec::fork_join(
                    ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC,
                    ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC,
                    ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC);

    using sndr_t = decltype(sndr);

    static_assert(std::same_as<
                  std::invoke_result_t<::stdexec::get_completion_signatures_t, sndr_t>,
                  ::stdexec::completion_signatures<::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>
    >);

    using clsr_t = ::stdexec::__clsur::__compose<
        ::stdexec::__closure<::stdexec::continues_on_t, Kokkos::Experimental::details::graph::Scheduler<execution_space>>,
        ::stdexec::__closure<::stdexec::then_t, ::tests::ThenFunctor<atomic<view_s_t>>>
    >;

    using outer_0 = ::stdexec::connect_result_t<sndr_t, ::tests::stdexec::SinkReceiver>;
    static_assert(std::same_as<
                  ::stdexec::__demangle_t<outer_0>,
                  Kokkos::Experimental::details::graph::ForkJoinOpState<
                      Kokkos::Experimental::details::graph::Scheduler<execution_space>,
                      Kokkos::Experimental::details::graph::Scheduler<execution_space>::Sender,
                      ::stdexec::__tuple<clsr_t, clsr_t, clsr_t>,
                      tests::stdexec::SinkReceiver
                  >
    >);

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { ::stdexec::sync_wait(std::move(sndr)); }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 3);
}

//! @test Use @c exec::fork_join with a diamond topology.
TEST_F(ForkJoinTest, diamond) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    using functor_t = ::tests::utils::LoadCheckAddFunctor<value_t, on_device>;

    auto sndr = ::stdexec::schedule(grc.get_scheduler())
              | ::stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | ::exec::fork_join(
                    ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC,
                    ::stdexec::continues_on(grc.get_scheduler()) | ADD_THEN_ATOMIC)
              | ::stdexec::then(functor_t{.prev = 6, .value = 3, .data = data.data()});

    using clsr_t = ::stdexec::__clsur::__compose<
        ::stdexec::__closure<::stdexec::continues_on_t, Kokkos::Experimental::details::graph::Scheduler<execution_space>>,
        ::stdexec::__closure<::stdexec::then_t, ::tests::ThenFunctor<atomic<view_s_t>>>
    >;

    using outer_0 = ::stdexec::connect_result_t<decltype(sndr), ::tests::stdexec::SinkReceiver>;
    static_assert(std::same_as<
                  ::stdexec::__demangle_t<outer_0>,
                  Kokkos::Experimental::details::graph::ThenOpState<
                      Kokkos::Experimental::details::graph::Scheduler<execution_space>,
                      ::stdexec::__basic_sender<
                          ::exec::fork_join_t,
                          ::stdexec::__tuple<clsr_t, clsr_t>,
                          ::stdexec::__basic_sender<
                              ::stdexec::then_t,
                              functor_t,
                              Kokkos::Experimental::details::graph::Scheduler<execution_space>::Sender
                          >
                      >,
                      tests::stdexec::SinkReceiver,
                      functor_t
                  >
    >);

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

/**
 * @test Use @c exec::fork_join after a @c stdexec::continues_on.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/issues/1823.
 */
TEST_F(ForkJoinTest, after_a_continues_on) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr =
        ::stdexec::just() | ::stdexec::continues_on(gctx.get_scheduler())
        | ::exec::fork_join(
            ::stdexec::continues_on(gctx.get_scheduler())
            | ::stdexec::then(
                ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 0, .value = 3, .data = data.data()}));

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
            ::stdexec::sync_wait(std::move(sndr));                       // NOLINT(performance-move-const-arg)
        }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 3);
}

//! @test Use @c exec::fork_join before a @c stdexec::continues_on.
TEST_F(ForkJoinTest, before_a_continues_on) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr =
        ::stdexec::schedule(gctx.get_scheduler())
        | ::exec::fork_join(
            ::stdexec::continues_on(gctx.get_scheduler())
            | ::stdexec::then(
                ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 0, .value = 3, .data = data.data()}))
        | ::stdexec::continues_on(gctx.get_scheduler())
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 3, .value = 3, .data = data.data()});

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(execution_space{}));
        }
    }
    matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
            ::stdexec::sync_wait(std::move(sndr));                       // NOLINT(performance-move-const-arg)
        }),
        ::testing::ElementsAreArray(matchers));

    ASSERT_EQ(data(), 6);
}

//! @test Nest @c exec::fork_join.
TEST_F(ForkJoinTest, nested) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t grc{exec};

    using functor_t = ::tests::utils::LoadCheckAddFunctor<value_t, on_device>;

    auto scheduler = grc.get_scheduler();

    auto sndr = ::stdexec::schedule(scheduler) | ::stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | ::exec::fork_join(
                    ::stdexec::continues_on(scheduler)
                        | ::exec::fork_join(
                            ::stdexec::continues_on(scheduler) | ADD_THEN_ATOMIC,
                            ::stdexec::continues_on(scheduler) | ADD_THEN_ATOMIC),
                    ::stdexec::continues_on(scheduler) | ADD_THEN_ATOMIC)
              | ::stdexec::then(functor_t{.prev = 7, .value = 5, .data = data.data()});

    using outer_0 = ::stdexec::connect_result_t<decltype(sndr), ::tests::stdexec::SinkReceiver>;
    static_assert(::stdexec::__is_instance_of<outer_0, Kokkos::Experimental::details::graph::ThenOpState>);

    static_assert(impl::check_node_type<
                  outer_0,
                  const impl::then_node_t<execution_space, functor_t, impl::aggregate_node_t<execution_space>>&
    >());

    using outer_1 = typename outer_0::inner_opstate_t;
    static_assert(::stdexec::__is_instance_of<outer_1, Kokkos::Experimental::details::graph::ForkJoinOpState>);

    static_assert(impl::check_node_type<outer_1, const impl::aggregate_node_t<execution_space>&>());

    std::vector<::testing::Matcher<variant_t>> matchers{
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
        MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
        KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

    if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
        if (execution_space{} != exec) {
            matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
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

    ASSERT_EQ(data(), 12);
}

} // namespace tests::kokkos_ext::graph
