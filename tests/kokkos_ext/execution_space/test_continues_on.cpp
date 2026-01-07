#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/Helpers.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/kokkos_ext/execution_space/Helpers.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c continues_on by @c Kokkos::Experimental::ExecutionSpaceContext
 * ----------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c continues_on.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_continues_on.cpp.
 */

using      execution_space = Kokkos::DefaultExecutionSpace;
using         memory_space = typename execution_space::memory_space;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace tests::kokkos_ext
{

using namespace Kokkos::utils::callbacks;

class ContinuesOnTest : public impl::ExecutionSpaceContextTest<execution_space, host_execution_space>,
                        public Kokkos::utils::tests::scoped::callbacks::Manager
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
    using variant_t           = std::variant    <BeginFenceEvent, BeginParallelForEvent>;
};

// https://github.com/NVIDIA/stdexec/blob/3363435259b7ffae43d3f2e5f6b7a7b36d7cd7d3/test/stdexec/algos/adaptors/test_continues_on.cpp#L227
TEST_F(ContinuesOnTest, completing_domain)
{
    const context_t esc{exec};

    ::stdexec::scheduler auto sched = esc.get_scheduler();

    ::stdexec::sender auto sndr = ::stdexec::continues_on(::stdexec::just(int{1}), sched);

    static_assert(std::same_as<
        ::stdexec::__domain_of_t<::stdexec::env_of_t<decltype(sndr)>>,
        ::stdexec::default_domain
    >);

    static_assert(std::same_as<
        ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, decltype(sndr)>,
        Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<execution_space>::Domain
    >);

    static_assert(std::same_as<
        ::stdexec::__completion_scheduler_of_t<::stdexec::set_value_t, decltype(sndr)>,
        int
    >);
}

/**
 * @test @c continues_on advertises the default domain, and completes on the @c exec::static_thread_pool domain,
 *       even when the chain is not started with a schedule sender.
 *
 * Similar to @ref tests::stdexec::adaptors::ContinuesOnTest_no_schedule_sender_continues_on_Test.
 */
TEST_F(ContinuesOnTest, no_schedule_sender_continues_on)
{
    const context_t esc{exec};

    ::stdexec::sender auto continues_on = ::stdexec::just() | ::stdexec::continues_on(esc.get_scheduler());

    using continues_on_t = decltype(continues_on);

    static_assert(std::same_as<
        ::stdexec::__domain_of_t<::stdexec::env_of_t<continues_on_t>>,
        ::stdexec::default_domain
    >);
    static_assert(std::same_as<
        ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, continues_on_t>,
        Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<execution_space>::Domain
    >);

    //! It also has a completion scheduler for the value channel.
    // static_assert(std::same_as<
    //     ::stdexec::__completion_scheduler_of_t<::stdexec::set_value_t, continues_on_t>,
    //     int
    // >);

    // ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    // ASSERT_THAT(
    //     recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
    //     ::testing::ElementsAre(
    //         MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
    //         MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
    //         MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))
    //     )
    // );

    // ASSERT_EQ(data(), 2) << "A synchronization is missing.";
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization)
 *       when using it many times on the same execution space instance.
 *
 * There shouldn't be any fencing required in this case.
 */
TEST_F(ContinuesOnTest, continues_on_noop)
{
    // const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    // const host_execution_space exec_h{};

    // const context_t esc{exec};

    // auto chain = ::stdexec::just()
    //     | ::stdexec::continues_on(esc.get_scheduler())
    //     | ADD_THEN
    //     | ::stdexec::continues_on(esc.get_scheduler())
    //     | ADD_THEN
    //     | ::stdexec::continues_on(esc.get_scheduler())
    //     | ADD_THEN;

    // using chain_0_t = decltype(chain_0);
    // using chain_1_t = decltype(chain_1);
    // using chain_2_t = decltype(chain_2);

    // static_assert(::stdexec::__has_completion_scheduler<chain_0_t, ::stdexec::set_value_t>);
    // static_assert(::stdexec::__has_completion_scheduler<chain_1_t, ::stdexec::set_value_t>);
    // static_assert(::stdexec::__has_completion_scheduler<chain_2_t, ::stdexec::set_value_t>);

    // static_assert(::stdexec::__completes_on<chain_0_t, scheduler_t>);
    // static_assert(::stdexec::__completes_on<chain_1_t, scheduler_h_t>);
    // static_assert(::stdexec::__completes_on<chain_2_t, scheduler_t>);

    // static_assert(tests::stdexec::has_completion_signatures<chain_0_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);
    // static_assert(tests::stdexec::has_completion_signatures<chain_1_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);
    // static_assert(tests::stdexec::has_completion_signatures<chain_2_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_0_t>, scheduler_domain_t>);
    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_1_t>, scheduler_domain_t>);
    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_2_t>, scheduler_domain_t>);

    // ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    // ASSERT_THAT(
    //     recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); }),
    //     ::testing::ElementsAre(
    //         MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
    //         MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
    //         MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
    //         MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))
    //     )
    // );

    // ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization) when using it many times.
 *
 * This is done by switching context at each @c then, hoping that at some point there will be a write race condition if
 * the synchronization is not properly implemented, thus failing the count test.
 */
TEST_F(ContinuesOnTest, continues_on_many)
{
    // const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    // const host_execution_space exec_h{};

    // const context_h_t esc_h{exec_h};
    // const context_t   esc  {exec};

    // std::cout << "--------- CHAIN 0 --------" << std::endl;
    // auto chain_0 = ::stdexec::just()
    //     | ::stdexec::continues_on(esc.get_scheduler()) // will be late customized
    //     | ADD_THEN;
    // std::cout << "--------- CHAIN 1 --------" << std::endl;
    // auto chain_1 = std::move(chain_0)
    //     | ::stdexec::continues_on(esc_h.get_scheduler())
    //     | ADD_THEN;
    // std::cout << "--------- CHAIN 2 --------" << std::endl;
    // auto chain_2 = std::move(chain_1)
    //     | ::stdexec::continues_on(esc.get_scheduler())
    //     | ADD_THEN;

    // using chain_0_t = decltype(chain_0);
    // using chain_1_t = decltype(chain_1);
    // using chain_2_t = decltype(chain_2);

    // static_assert(::stdexec::__has_completion_scheduler<chain_0_t, ::stdexec::set_value_t>);
    // static_assert(::stdexec::__has_completion_scheduler<chain_1_t, ::stdexec::set_value_t>);
    // static_assert(::stdexec::__has_completion_scheduler<chain_2_t, ::stdexec::set_value_t>);

    // static_assert(::stdexec::__completes_on<chain_0_t, scheduler_t>);
    // static_assert(::stdexec::__completes_on<chain_1_t, scheduler_h_t>);
    // static_assert(::stdexec::__completes_on<chain_2_t, scheduler_t>);

    // static_assert(tests::stdexec::has_completion_signatures<chain_0_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);
    // static_assert(tests::stdexec::has_completion_signatures<chain_1_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);
    // static_assert(tests::stdexec::has_completion_signatures<chain_2_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_0_t>, scheduler_domain_t>);
    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_1_t>, scheduler_h_domain_t>);
    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_2_t>, scheduler_domain_t>);

    // ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    // std::cout << "> exec  : " << Kokkos::Tools::Experimental::device_id(exec) << std::endl;
    // std::cout << "> exec_h: " << Kokkos::Tools::Experimental::device_id(exec_h) << std::endl;

    // const auto recorded_events = recorder_listener_t::record([chain = std::move(chain_2)] () mutable { ::stdexec::sync_wait(std::move(chain)); });
    // for (const auto& recorded_event : recorded_events) {
    //     std::visit([] (const auto& arg) { std::cout << "- " << arg << std::endl; }, recorded_event);
    // }

    // ASSERT_THAT(
    //     recorded_events,
    //     ::testing::ElementsAre(
    //         MATCHER_FOR_BEGIN_PFOR (exec,   dispatch_label("then")),
    //         MATCHER_FOR_BEGIN_FENCE(exec,   dispatch_label("schedule_from")),
    //         MATCHER_FOR_BEGIN_PFOR (exec_h, dispatch_label("then")),
    //         MATCHER_FOR_BEGIN_FENCE(exec_h, dispatch_label("schedule_from")),
    //         MATCHER_FOR_BEGIN_PFOR (exec,   dispatch_label("then")),
    //         MATCHER_FOR_BEGIN_FENCE(exec,   dispatch_label("sync_wait"))
    //     )
    // );

    // ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

TEST_F(ContinuesOnTest, stupidous)
{
    // const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    // const host_execution_space exec_h{};

    // const context_h_t esc_h{exec_h};
    // const context_t   esc  {exec};

    // std::cout << "--------- CHAIN 0 --------" << std::endl;
    // auto chain_0 = ::stdexec::schedule(esc.get_scheduler())
    //     | ADD_THEN;
    // std::cout << "--------- CHAIN 1 --------" << std::endl;
    // auto chain_1 = std::move(chain_0)
    //     | ::stdexec::continues_on(esc_h.get_scheduler())
    //     | ADD_THEN;
    // std::cout << "--------- CHAIN 2 --------" << std::endl;
    // auto chain_2 = std::move(chain_1)
    //     | ::stdexec::continues_on(esc.get_scheduler())
    //     | ADD_THEN;

    // using chain_0_t = decltype(chain_0);
    // using chain_1_t = decltype(chain_1);
    // using chain_2_t = decltype(chain_2);

    // static_assert(::stdexec::__has_completion_scheduler<chain_0_t, ::stdexec::set_value_t>);
    // static_assert(::stdexec::__has_completion_scheduler<chain_1_t, ::stdexec::set_value_t>);
    // static_assert(::stdexec::__has_completion_scheduler<chain_2_t, ::stdexec::set_value_t>);

    // static_assert(::stdexec::__completes_on<chain_0_t, scheduler_t>);
    // static_assert(::stdexec::__completes_on<chain_1_t, scheduler_h_t>);
    // static_assert(::stdexec::__completes_on<chain_2_t, scheduler_t>);

    // static_assert(tests::stdexec::has_completion_signatures<chain_0_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);
    // static_assert(tests::stdexec::has_completion_signatures<chain_1_t, ::stdexec::set_value_t(), ::stdexec::set_error_t(std::exception_ptr)>);
    // static_assert(tests::stdexec::has_completion_signatures<chain_2_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_value_t()>);

    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_0_t>, scheduler_domain_t>);
    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_1_t>, scheduler_h_domain_t>);
    // static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_2_t>, scheduler_domain_t>);

    // ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    // std::cout << "> exec  : " << Kokkos::Tools::Experimental::device_id(exec) << std::endl;
    // std::cout << "> exec_h: " << Kokkos::Tools::Experimental::device_id(exec_h) << std::endl;

    // const auto recorded_events = recorder_listener_t::record([chain = std::move(chain_2)] () mutable { ::stdexec::sync_wait(std::move(chain)); });
    // for (const auto& recorded_event : recorded_events) {
    //     std::visit([] (const auto& arg) { std::cout << "- " << arg << std::endl; }, recorded_event);
    // }

    // ASSERT_THAT(
    //     recorded_events,
    //     ::testing::ElementsAre(
    //         MATCHER_FOR_BEGIN_PFOR (exec,   then),
    //         MATCHER_FOR_BEGIN_FENCE(exec,   schedule_from),
    //         MATCHER_FOR_BEGIN_PFOR (exec_h, then),
    //         MATCHER_FOR_BEGIN_FENCE(exec_h, schedule_from),
    //         MATCHER_FOR_BEGIN_PFOR (exec,   then),
    //         MATCHER_FOR_BEGIN_FENCE(exec,   sync_wait)
    //     )
    // );

    // ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

} // namespace tests::kokkos_ext
