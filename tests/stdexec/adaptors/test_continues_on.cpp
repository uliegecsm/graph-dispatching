#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"

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

/**
 * @test Simple test for @c stdexec::continues_on.
 *
 * The beginning of the chain is scheduled on one execution resource, and the flow is then
 * transferred to another execution resource.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/test/execpools/test_taskflow_thread_pool.cpp#L122.
 */
TEST(stdexec, continues_on)
{
    exec::static_thread_pool pool_A{1}, pool_B{1};

    ::stdexec::scheduler auto scheduler_A = pool_A.get_scheduler();
    ::stdexec::scheduler auto scheduler_B = pool_B.get_scheduler();

    std::thread::id thr_A, thr_B;
    size_t counter = 0;

    auto chain = ::stdexec::schedule(scheduler_A)
        | ::stdexec::then([&]() -> void { thr_A = std::this_thread::get_id(); ++counter; })
        | ::stdexec::continues_on(scheduler_B)
        | ::stdexec::then([&]() -> void { thr_B = std::this_thread::get_id(); ++counter; });

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_NE(thr_A, thr_B);
    ASSERT_EQ(counter, 2);
}

//! @test This test checks that using @c continues_on persists the scheduler.
TEST(stdexec, continues_on_persists_scheduler)
{
    exec::static_thread_pool pool_a{1}, pool_b{1};

    auto sch_a = pool_a.get_scheduler();
    auto sch_b = pool_b.get_scheduler();

    //! The schedulers are not equal.
    ASSERT_NE(sch_a, sch_b);

    const std::thread::id thr_a = std::get<0>(::stdexec::sync_wait(::stdexec::schedule(sch_a) | ::stdexec::then([]{ return std::this_thread::get_id(); })).value());
    const std::thread::id thr_b = std::get<0>(::stdexec::sync_wait(::stdexec::schedule(sch_b) | ::stdexec::then([]{ return std::this_thread::get_id(); })).value());

    //! Each scheduler has a unique thread.
    ASSERT_NE(thr_a, thr_b);

    #define THEN_GET_THR(__var__) ::stdexec::then([&__var__]{ __var__ = std::this_thread::get_id(); })

    #define CHECK_COMPLETION_SIGNATURES(__who__, ...)                                                                              \
        using __who__##_completion_signatures_t = std::invoke_result_t<::stdexec::get_completion_signatures_t, decltype(__who__)>; \
        static_assert(std::same_as<__who__##_completion_signatures_t, ::stdexec::completion_signatures<__VA_ARGS__>>);

    //! First then, completion on scheduler 'a'.
    std::thread::id thr_1_on_a;
    auto then_1_on_a = ::stdexec::schedule(sch_a) | THEN_GET_THR(thr_1_on_a);
    ASSERT_EQ(sch_a, ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(then_1_on_a)));

    CHECK_COMPLETION_SIGNATURES(then_1_on_a, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_stopped_t(), ::stdexec::set_value_t());

    //! Next then, still on scheduler 'a'.
    std::thread::id thr_2_on_a;
    auto then_2_on_a = std::move(then_1_on_a) | THEN_GET_THR(thr_2_on_a);
    ASSERT_EQ(sch_a, ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(then_2_on_a)));

    CHECK_COMPLETION_SIGNATURES(then_2_on_a, ::stdexec::set_value_t(), ::stdexec::set_stopped_t(), ::stdexec::set_error_t(std::exception_ptr));

    //! Now, we move on scheduler 'b'.
    std::thread::id thr_1_on_b;
    auto then_1_on_b = std::move(then_2_on_a) | ::stdexec::continues_on(sch_b) | THEN_GET_THR(thr_1_on_b);
    ASSERT_EQ(sch_b, ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(then_1_on_b)));

    CHECK_COMPLETION_SIGNATURES(then_1_on_b, ::stdexec::set_value_t(), ::stdexec::set_stopped_t(), ::stdexec::set_error_t(std::exception_ptr));

    //! The second then is still on scheduler 'b'.
    std::thread::id thr_2_on_b;
    auto then_2_on_b = std::move(then_1_on_b) | THEN_GET_THR(thr_2_on_b);
    ASSERT_EQ(sch_b, ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(then_2_on_b)));

    CHECK_COMPLETION_SIGNATURES(then_2_on_b, ::stdexec::set_error_t(std::exception_ptr), ::stdexec::set_stopped_t(), ::stdexec::set_value_t());

    ::stdexec::sync_wait(std::move(then_2_on_b));

    //! Check which thread got the work.
    ASSERT_EQ(thr_a, thr_1_on_a); ASSERT_EQ(thr_b, thr_1_on_b);
    ASSERT_EQ(thr_a, thr_2_on_a); ASSERT_EQ(thr_b, thr_2_on_b);
}

} // namespace tests::stdexec::adaptors
