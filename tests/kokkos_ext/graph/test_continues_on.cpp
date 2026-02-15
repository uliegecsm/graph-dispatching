#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/Utils.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/graph/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/LoadCheckAdd.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c continues_on by @c Kokkos::Experimental::GraphContext
 * -------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::GraphContext properly customizes
 * @c continues_on.
 *
 * The tests can be found in @ref tests/kokkos_ext/graph/test_continues_on.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class ContinuesOnTest
    : public impl::GraphContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent, ProfileEvent>;

    static constexpr bool on_device = ::tests::utils::on_device<execution_space>();
};

//! @test Check traits of the sender created by the customized @c continues_on.
TEST_F(ContinuesOnTest, traits) {
    static_assert(::utils::check_continues_on<decltype(context_t{exec}.get_scheduler())>());
}

//! @test A @c then and a @c sync_wait following a @c continues_on properly use the graph.
TEST_F(ContinuesOnTest, then_sync_wait) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    ::stdexec::sender auto chain = ::stdexec::just() | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test Check that @c continues_on is properly customized when using it many times on the same graph.
 *
 * All @ref Kokkos::Experimental::details::graph::ThenOpState add their node to the same graph.
 */
TEST_F(ContinuesOnTest, transition_to_same_graph) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    using functor_t = tests::utils::LoadCheckAddFunctor<value_t, on_device>;

    auto chain = ::stdexec::just() | ::stdexec::continues_on(esc.get_scheduler())
               | ::stdexec::then(functor_t{.prev = 0, .value = 1, .data = data.data()})
               | ::stdexec::continues_on(esc.get_scheduler())
               | ::stdexec::then(functor_t{.prev = 1, .value = 2, .data = data.data()})
               | ::stdexec::continues_on(esc.get_scheduler())
               | ::stdexec::then(functor_t{.prev = 3, .value = 3, .data = data.data()});

    /// Since all operation states that create a node are instances of @ref Kokkos::Experimental::details::graph::ThenOpState
    /// or @ref Kokkos::Experimental::details::graph::ContinuesOnOpState,
    /// they are able to retrieve the node from their inner operation state and the resulting @c Kokkos graph is correct.
    using outer_0 = ::stdexec::connect_result_t<decltype(chain), ::tests::stdexec::SinkReceiver>;
    static_assert(::stdexec::__is_instance_of<outer_0, Kokkos::Experimental::details::graph::ThenOpState>);

    using outer_1 = typename outer_0::inner_opstate_t;
    static_assert(::stdexec::__is_instance_of<outer_1, Kokkos::Experimental::details::graph::ContinuesOnOpState>);

    using outer_2 = typename outer_1::inner_opstate_t;
    static_assert(::stdexec::__is_instance_of<outer_2, Kokkos::Experimental::details::graph::ThenOpState>);

    using outer_3 = typename outer_2::inner_opstate_t;
    static_assert(::stdexec::__is_instance_of<outer_3, Kokkos::Experimental::details::graph::ContinuesOnOpState>);

    using outer_4 = typename outer_3::inner_opstate_t;
    static_assert(::stdexec::__is_instance_of<outer_4, Kokkos::Experimental::details::graph::ThenOpState>);

    using outer_5 = typename outer_4::inner_opstate_t;
    static_assert(::stdexec::__is_instance_of<outer_5, Kokkos::Experimental::details::graph::ContinuesOnOpState>);

    using outer_6 = typename outer_5::inner_opstate_t;
    static_assert(::stdexec::__is_instance_of<outer_6, ::stdexec::__opstate>);

    using then_a_t = impl::then_node_t<execution_space, functor_t, impl::root_node_t<execution_space>>;
    using then_b_t = impl::then_node_t<execution_space, functor_t, then_a_t>;
    using then_c_t = impl::then_node_t<execution_space, functor_t, then_b_t>;

    static_assert(impl::check_node_type<outer_5, impl::root_node_t<execution_space>>());
    static_assert(impl::check_node_type<outer_4, const then_a_t&>());
    static_assert(impl::check_node_type<outer_3, const then_a_t&>());
    static_assert(impl::check_node_type<outer_2, const then_b_t&>());
    static_assert(impl::check_node_type<outer_1, const then_b_t&>());
    static_assert(impl::check_node_type<outer_0, const then_c_t&>());

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { // NOLINT(performance-move-const-arg)
            ::stdexec::sync_wait(std::move(chain));                        // NOLINT(performance-move-const-arg)
        }),
        ::testing::ElementsAre(
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 6) << "A synchronization is missing.";
}

} // namespace tests::kokkos_ext
