#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/graph/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/LoadCheckAdd.hpp"
#include "tests/utils/ThrowsWhenCopied.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c then by @c Kokkos::Experimental::GraphContext
 * -----------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::GraphContext properly customizes
 * @c then.
 *
 * The tests can be found in @ref tests/kokkos_ext/graph/test_then.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class ThenTest
    : public impl::GraphContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<BeginFenceEvent, BeginParallelForEvent, AllocateDataEvent, DeallocateDataEvent, ProfileEvent>;

    using value_t = typename view_s_t::value_type;

    static constexpr bool on_device = ::tests::utils::on_device<execution_space>();
};

/**
 * @test Check that @ref Kokkos::Experimental::GraphContext does its duty well when used with @c then
 *       within a chain started with @c stdexec::schedule.
 */
TEST_F(ThenTest, then_schedule) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain =
        ::stdexec::schedule(esc.get_scheduler())
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 0, .value = 4, .data = data.data()})
        | ::stdexec::then(
            ::tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 4, .value = 2, .data = data.data()});

    using chain_t = decltype(chain);

    //! The chain environment advertises the default domain, and completes on the @ref Kokkos::Experimental::details::graph::Domain domain.
    static_assert(std::same_as<::stdexec::__domain_of_t<::stdexec::env_of_t<chain_t>>, ::stdexec::default_domain>);
    static_assert(std::same_as<
                  ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, chain_t>,
                  Kokkos::Experimental::details::graph::Domain
    >);

    //! It has a completion scheduler for the value channel.
    static_assert(std::same_as<
                  decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(chain))),
                  scheduler_t
    >);

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { // NOLINT(performance-move-const-arg)
            ::stdexec::sync_wait(std::move(chain));                        // NOLINT(performance-move-const-arg)
        }),
        ::testing::ElementsAre(
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 6);
}

/**
 * @test Similar to @ref tests::kokkos_ext::ThenTest_then_schedule_Test, but the chain is scheduled
 *       with a @c starts_on.
 */
TEST_F(ThenTest, then_starts_on) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    //! Create a chain that does not start with a schedule sender.
    auto chain = ::stdexec::just() | ADD_THEN;

    /// The chain cannot be queried for a completion scheduler.
    /// It may complete on the value channel or the error channel, since @c Kokkos may throw.
    /// It hasn't been connected yet, so the domain is indeterminate.
    using chain_t = decltype(chain);

    static_assert(!tests::stdexec::has_completion_scheduler_for<chain_t, ::stdexec::set_value_t>);
    static_assert(tests::stdexec::has_completion_signatures<
                  chain_t,
                  ::stdexec::set_error_t(std::exception_ptr),
                  ::stdexec::set_value_t()
    >);

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, chain_t>,
                  ::stdexec::indeterminate_domain<>
    >);

    //! Call @c starts_on.
    auto starts_on = ::stdexec::starts_on(esc.get_scheduler(), std::move(chain));

    using starts_on_t = decltype(starts_on);

    //! It has a completion scheduler for the value channel.
    static_assert(tests::stdexec::has_completion_scheduler_for<starts_on_t, ::stdexec::set_value_t>);
    static_assert(std::same_as<
                  decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(starts_on))),
                  scheduler_t
    >);

    static_assert(std::same_as<
                  std::invoke_result_t<::stdexec::get_completion_signatures_t, starts_on_t>,
                  ::stdexec::completion_signatures<::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>
    >);

    static_assert(std::same_as<
                  std::invoke_result_t<::stdexec::get_completion_signatures_t, starts_on_t, ::stdexec::env<>>,
                  ::stdexec::completion_signatures<::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record(
            [starts_on = std::move(starts_on)]() mutable { ::stdexec::sync_wait(std::move(starts_on)); }),
        ::testing::ElementsAre(
            MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
            KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

//! @test If an exception is thrown while dispatching a @c Kokkos parallel region, it is properly caught and propagated.
TEST_F(ThenTest, error_propagates) {
    const context_t esc{exec};

    ::stdexec::sender auto sndr = ::stdexec::schedule(esc.get_scheduler())
                                | ::stdexec::then(::tests::utils::ThrowsWhenCopied{});

    ASSERT_THAT(
        ::tests::utils::MutableMoveToSyncWait{.sndr = std::move(sndr)},
        ::testing::ThrowsMessage<std::runtime_error>(
            testing::HasSubstr("ThrowsWhenCopied: Throwing in copy constructor!")));
}

} // namespace tests::kokkos_ext
