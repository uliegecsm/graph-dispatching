#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::continues_on
 * ----------------------------------
 *
 * This group of tests check the behavior of @c stdexec::continues_on.
 *
 * The test can be found in @ref stdexec/adaptors/test_continues_on.cpp.
 */

namespace tests::stdexec::adaptors
{

class ContinuesOnTest : public utils::StaticThreadPool<'A', 'B'>, public ::testing::Test
{
public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
};

/**
 * @test Simple test for @c stdexec::continues_on.
 *
 * The beginning of the chain is scheduled on one execution resource, and the flow is then
 * transferred to another execution resource.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/test/execpools/test_taskflow_thread_pool.cpp#L122.
 */
TEST_F(ContinuesOnTest, continues_on)
{
    std::array<std::thread::id, 3> thrids;
    size_t counter = 0;

    auto chain = ::stdexec::schedule(pools.at(index_of_A).get_scheduler())
        | THEN_STORE_ID(thrids[0], {++counter;})
        | ::stdexec::continues_on(pools.at(index_of_B).get_scheduler())
        | THEN_STORE_ID(thrids[1], {++counter;})
        | THEN_STORE_ID(thrids[2], {++counter;});

    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(thrids, ::testing::ElementsAre(
        threads.at(index_of_A),
        threads.at(index_of_B),
        threads.at(index_of_B)));
    ASSERT_EQ(counter, 3);
}

/**
 * @test @c continues_on advertises the default domain, and completes on the @c exec::static_thread_pool domain,
 *       even when the chain is not started with a schedule sender.
 */
TEST_F(ContinuesOnTest, no_schedule_sender_continues_on)
{
    auto continues_on = ::stdexec::just() | ::stdexec::continues_on(pools.at(index_of_A).get_scheduler());

    using continues_on_t = decltype(continues_on);

    static_assert(std::same_as<
        ::stdexec::__domain_of_t<::stdexec::env_of_t<continues_on_t>>,
        ::stdexec::default_domain
    >);
    static_assert(std::same_as<
        ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, continues_on_t>,
        ::exec::_pool_::_static_thread_pool::domain
    >);

    //! It also has a completion scheduler for the value channel.
    static_assert(std::same_as<
        ::stdexec::__completion_scheduler_of_t<::stdexec::set_value_t, continues_on_t>,
        ::exec::_pool_::_static_thread_pool::scheduler
    >);
}

//! @test This test checks that using @c continues_on persists the scheduler.
TEST_F(ContinuesOnTest, continues_on_persists_scheduler)
{
    auto sch_a = pools.at(index_of_A).get_scheduler();
    auto sch_b = pools.at(index_of_B).get_scheduler();

    //! The schedulers are not equal.
    ASSERT_NE(sch_a, sch_b);

    //! Each scheduler has a unique thread.
    ASSERT_NE(threads.at(index_of_A), threads.at(index_of_B));

    //! First then, completion on scheduler 'a'.
    std::thread::id thr_1_on_a;
    auto then_1_on_a = ::stdexec::schedule(sch_a) | THEN_STORE_ID(thr_1_on_a);
    ASSERT_EQ(sch_a, ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(then_1_on_a)));

    static_assert(has_completion_signatures<decltype(then_1_on_a), ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_stopped_t(), ::stdexec::set_value_t()>);

    //! Next then, still on scheduler 'a'.
    std::thread::id thr_2_on_a;
    auto then_2_on_a = std::move(then_1_on_a) | THEN_STORE_ID(thr_2_on_a); // NOLINT(performance-move-const-arg)
    ASSERT_EQ(sch_a, ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(then_2_on_a)));

    static_assert(has_completion_signatures<decltype(then_2_on_a), ::stdexec::set_value_t(), ::stdexec::set_stopped_t(), ::stdexec::set_error_t(std::exception_ptr)>);

    /// Now, we move on scheduler 'b'.
    auto continues_on = std::move(then_2_on_a) | ::stdexec::continues_on(sch_b); // NOLINT(performance-move-const-arg)

    //! Let's perform some traits checks on the chain from the @c continues_on.
    using continues_on_t = decltype(continues_on);

    /// @c continues_on advertises the default domain, and completes on the @c exec::static_thread_pool domain.
    static_assert(std::same_as<
        ::stdexec::__domain_of_t<::stdexec::env_of_t<continues_on_t>>,
        ::stdexec::default_domain
    >);
    static_assert(std::same_as<
        ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, continues_on_t>,
        ::exec::_pool_::_static_thread_pool::domain
    >);

    //! It also has a completion scheduler for the value channel.
    static_assert(std::same_as<
        ::stdexec::__completion_scheduler_of_t<::stdexec::set_value_t, continues_on_t>,
        exec::_pool_::_static_thread_pool::scheduler
    >);

    //! First then on scheduler 'b'.
    std::thread::id thr_1_on_b;
    auto then_1_on_b = std::move(continues_on) | THEN_STORE_ID(thr_1_on_b); // NOLINT(performance-move-const-arg)
    ASSERT_EQ(sch_b, ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(then_1_on_b)));

    using then_1_on_b_t = decltype(then_1_on_b);

    static_assert(has_completion_signatures<then_1_on_b_t, ::stdexec::set_value_t(), ::stdexec::set_stopped_t(), ::stdexec::set_error_t(std::exception_ptr)>);

    //! It advertises the default domain, and completes on the @c exec::static_thread_pool domain.
    static_assert(std::same_as<
        ::stdexec::__domain_of_t<::stdexec::env_of_t<then_1_on_b_t>>,
        ::stdexec::default_domain
    >);
    static_assert(std::same_as<
        ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, then_1_on_b_t>,
        ::exec::_pool_::_static_thread_pool::domain
    >);

    //! The second then is still on scheduler 'b'.
    std::thread::id thr_2_on_b;
    auto then_2_on_b = std::move(then_1_on_b) | THEN_STORE_ID(thr_2_on_b); // NOLINT(performance-move-const-arg)
    ASSERT_EQ(sch_b, ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(then_2_on_b)));

    using then_2_on_b_t = decltype(then_2_on_b);

    static_assert(has_completion_signatures<then_2_on_b_t, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_stopped_t(), ::stdexec::set_value_t()>);

    //! It advertises the default domain, and completes on the @c exec::static_thread_pool domain.
    static_assert(std::same_as<
        ::stdexec::__domain_of_t<::stdexec::env_of_t<then_2_on_b_t>>,
        ::stdexec::default_domain
    >);
    static_assert(std::same_as<
        ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, then_2_on_b_t>,
        ::exec::_pool_::_static_thread_pool::domain
    >);

    ::stdexec::sync_wait(std::move(then_2_on_b)); // NOLINT(performance-move-const-arg)

    //! Check which thread got the work.
    ASSERT_EQ(threads.at(index_of_A), thr_1_on_a); ASSERT_EQ(threads.at(index_of_B), thr_1_on_b);
    ASSERT_EQ(threads.at(index_of_A), thr_2_on_a); ASSERT_EQ(threads.at(index_of_B), thr_2_on_b);
}

} // namespace tests::stdexec::adaptors
