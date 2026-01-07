#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::split
 * ---------------------------
 *
 * This group of tests check the behavior of @c stdexec::split, and therefore of @c stdexec::when_all as well.
 *
 * The test can be found in @ref stdexec/adaptors/test_split.cpp.
 */

namespace tests::stdexec::adaptors
{

//! @test Simple test for @c stdexec::split.
TEST(stdexec, split)
{
    exec::static_thread_pool pool{2};

    ::stdexec::scheduler auto scheduler = pool.get_scheduler();

    std::array<std::thread::id, 2> bulk_thr;
    std::thread::id thr_A, thr_B;

    auto start = ::stdexec::schedule(scheduler)
        | ::stdexec::bulk(::stdexec::par, 2, [&](const auto index) -> void { bulk_thr[index] = std::this_thread::get_id(); })
        | ::stdexec::split();

    auto node_A =           start  | ::stdexec::then([&] { thr_A = std::this_thread::get_id(); });
    auto node_B = std::move(start) | ::stdexec::then([&] { thr_B = std::this_thread::get_id(); });

    auto chain = ::stdexec::when_all(std::move(node_A), std::move(node_B));

    ::stdexec::sync_wait(std::move(chain));

    //! Each @c bulk work item has been processed by a different thread.
    ASSERT_NE(bulk_thr[0], bulk_thr[1]);

    //! However, both asynchronous @c then have been run by the same thread.
    ASSERT_EQ(thr_A, thr_B);
}

class SplitTest : public utils::StaticThreadPool<'A', 'B', 'C'>, public ::testing::Test
{
public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
    static constexpr size_t index_of_C = index_of<'C'>();

    void test_with_transition()
    {
        ::stdexec::scheduler auto scheduler_A = this->pools.at(index_of_A).get_scheduler();
        ::stdexec::scheduler auto scheduler_B = this->pools.at(index_of_B).get_scheduler();
        ::stdexec::scheduler auto scheduler_C = this->pools.at(index_of_C).get_scheduler();

        std::array<std::thread::id, 9> thrids;

        //! The starting chain has a completion scheduler already.
        ::stdexec::sender auto start = ::stdexec::schedule(scheduler_A) | THEN_STORE_ID(thrids[0]);

        static_assert(std::same_as<
            ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(start)>,
            ::exec::_pool_::_static_thread_pool::domain
        >);
        static_assert(has_completion_scheduler_for<decltype(start), ::stdexec::set_value_t>);

        //! However, once @c stdexec::split is used, there is no completion scheduler anymore.
        ::stdexec::sender auto fork_one = std::move(start) | ::stdexec::split(); // NOLINT(performance-move-const-arg)

        static_assert(std::same_as<
            ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(fork_one)>,
            ::stdexec::indeterminate_domain<>
        >);
        static_assert(!has_completion_scheduler_for<decltype(fork_one), ::stdexec::set_value_t>);

        PRAGMA_DIAGNOSTIC_PUSH
        PRAGMA_DIAGNOSTIC_IGNORED_STRINGOP_OVERFLOW
        ::stdexec::sender auto stage_one = ::stdexec::when_all(
                    fork_one  | THEN_STORE_ID(thrids[1]),
            std::move(fork_one) | THEN_STORE_ID(thrids[2])
        );
        PRAGMA_DIAGNOSTIC_POP

        static_assert(std::same_as<
            ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(stage_one)>,
            ::stdexec::indeterminate_domain<>
        >);
        static_assert(!has_completion_scheduler_for<decltype(stage_one), ::stdexec::set_value_t>);

        auto post_one = std::move(stage_one) | THEN_STORE_ID(thrids[3]);

        static_assert(std::same_as<
            ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(post_one), ::stdexec::env<>>,
            ::stdexec::default_domain
        >);
        static_assert(!has_completion_scheduler_for<decltype(post_one), ::stdexec::set_value_t>);

        auto fork_two = std::move(post_one) | ::stdexec::continues_on(scheduler_B) | THEN_STORE_ID(thrids[4]) | ::stdexec::split();

        PRAGMA_DIAGNOSTIC_PUSH
        PRAGMA_DIAGNOSTIC_IGNORED_STRINGOP_OVERFLOW
        auto stage_two = ::stdexec::when_all(
                    fork_two | ::stdexec::continues_on(scheduler_C) | THEN_STORE_ID(thrids[5]),
                    fork_two | ::stdexec::continues_on(scheduler_A) | THEN_STORE_ID(thrids[6]),
            std::move(fork_two)                                       | THEN_STORE_ID(thrids[7])
        );
        PRAGMA_DIAGNOSTIC_POP

        //! The execution context of the trailing sender is unknown.
        auto post_two = std::move(stage_two) | THEN_STORE_ID(thrids[8]);

        ::stdexec::sync_wait(std::move(post_two));

        std::cout << "> A: " << threads.at(index_of_A) << std::endl;
        std::cout << "> B: " << threads.at(index_of_B) << std::endl;
        std::cout << "> C: " << threads.at(index_of_C) << std::endl;
        std::cout << ">  : " << std::this_thread::get_id() << std::endl;

        ASSERT_THAT(thrids, ::testing::ElementsAre(
            threads.at(index_of_A),
            threads.at(index_of_A),
            threads.at(index_of_A),
            threads.at(index_of_A),
            threads.at(index_of_B),
            threads.at(index_of_C),
            threads.at(index_of_A),
            threads.at(index_of_B),
            ::testing::AnyOf(threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_C))
        ));
    }
};

//! @test The sender returned by @c stdexec::split or @c stdexec::when_all does not have a completion scheduler.
TEST_F(SplitTest, with_transition)
{
    //! GCC 13.3.0 and 14.2.0 seem to raise a false positive for -Wstringop-overflow, but it will segfault.
#if defined(__GNUC__) && !defined(__clang__)
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH(this->test_with_transition(), ".*");
#else
    this->test_with_transition();
#endif
}

} // namespace tests::stdexec::adaptors
