#include <vector>

#include "gmock/gmock.h"
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
 * Tests for @c stdexec::starts_on
 * -------------------------------
 *
 * This group of tests check the behavior of @c stdexec::starts_on.
 *
 * The test can be found in @ref test_starts_on.cpp.
 */

namespace tests::stdexec::adaptors
{

/// @test Simple @c stdexec::starts_on test, that builds a chain from @c stdexec::just and starts it on two distinct schedulers.
/// @note We therefore use a @c multi-shot chain (we can consume it several times).
TEST(stdexec, starts_on_twice_with_just_a_bulk)
{
    constexpr size_t size = 4;

    //! Create 2 thread pools (A and B) with one thread each.
    exec::static_thread_pool pool_A{1}, pool_B{1};

    //! Get a scheduler from the both thread pools.
    ::stdexec::scheduler auto scheduler_A = pool_A.get_scheduler();
    ::stdexec::scheduler auto scheduler_B = pool_B.get_scheduler();

    /// The workload will fill a vector with the thread ID that procecesses the work item.
    /// Note that we will create the chain only once, and execute it twice (once on each scheduler).
    /// Also note that @c stdexec::just will decay-copied the received value, and in this case will pass a copy
    /// to the connected receiver.
    ::stdexec::sender auto chain = ::stdexec::just(std::vector<size_t>(size, 0))
        | ::stdexec::bulk(
            size, [](const auto index, auto& data) {
                data[index] = ::utils::get_thread_id();
    });

    //! Run on pool A.
    ::stdexec::sender auto moved_to_another_A = ::stdexec::start_on(scheduler_A, chain);
    const auto [result_A] = ::stdexec::sync_wait(std::move(moved_to_another_A)).value();

    //! Run on pool B.
    ::stdexec::sender auto moved_to_another_B = ::stdexec::start_on(scheduler_B, chain);
    const auto [result_B] = ::stdexec::sync_wait(std::move(moved_to_another_B)).value();

    //! Since we used @c stdexec::just, we expect that each "executed chain" produced its own @c std::vector.
    ASSERT_EQ(result_A.size(), size);
    ASSERT_EQ(result_A.size(), result_B.size());
    ASSERT_NE(result_A.data(), result_B.data());

    /// Each thread pool contains a different thread (not sure why), but all the work of a given chain is executed by the same thread
    /// since the thread pool since is one.
    ASSERT_NE(result_A.at(0), result_B.at(0));

    ASSERT_THAT(result_A, ::testing::Each(result_A.at(0)));
    ASSERT_THAT(result_B, ::testing::Each(result_B.at(0)));
}

/**
 * @test This test shows that if a chain is started using a @c stdexec::schedule
 *       sender, then using @c stdexec::start_on later on with another @c stdexec::schedule
 *       sender is a no-op.
 */
TEST(stdexec, starts_on_B_once_after_schedule_on_A_is_a_no_op)
{
    constexpr size_t size = 4;

    //! Create 2 thread pools (A and B) with one thread each.
    exec::static_thread_pool pool_A{1}, pool_B{1};

    //! Get a scheduler from the both thread pools.
    ::stdexec::scheduler auto scheduler_A = pool_A.get_scheduler();
    ::stdexec::scheduler auto scheduler_B = pool_B.get_scheduler();

    //! Collect the thread ID in each pool.
    const auto [pool_A_thread_ID] = ::stdexec::sync_wait(
        ::stdexec::schedule(scheduler_A) | ::stdexec::then([]() -> size_t { return ::utils::get_thread_id(); } )).value();

    const auto [pool_B_thread_ID] = ::stdexec::sync_wait(
        ::stdexec::schedule(scheduler_B) | ::stdexec::then([]() -> size_t { return ::utils::get_thread_id(); } )).value();

    ASSERT_NE(pool_A_thread_ID, 0);
    ASSERT_NE(pool_B_thread_ID, 0);
    ASSERT_NE(pool_A_thread_ID, pool_B_thread_ID);

    /// The workload will fill a vector with the thread ID that processes the work item.
    /// We start the chain with a schedule on pool A, and then try to
    /// start it on pool B, such that we could expect that the vector will be filled
    /// with the ID of the thread in pool B. But the chain will not be taken care of by the pool B
    /// but by pool A, since we started the chain with it.
    std::vector<size_t> data(size, 0);

    ::stdexec::sender auto chain = ::stdexec::schedule(scheduler_A)
        | ::stdexec::bulk(
            size, [&](const auto index) {
                data[index] = ::utils::get_thread_id();
    });

    ::stdexec::sync_wait(::stdexec::start_on(scheduler_B, chain));

    ASSERT_THAT(data, ::testing::Each(pool_A_thread_ID));
}

} // namespace tests::stdexec::adaptors
