#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED_DEPRECATED_ATTRIBUTES
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "exec/split.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::split
 * ---------------------------
 *
 * This group of tests check the behavior of @c exec::split, and therefore of @c stdexec::when_all as well.
 *
 * The test can be found in @ref tests/exec/adaptors/test_split.cpp.
 */

namespace tests::stdexec::adaptors {

//! @test Simple test for @c exec::split.
TEST(stdexec, split) {
    exec::static_thread_pool pool{2}; // NOLINT(misc-const-correctness)

    ::stdexec::scheduler auto scheduler = pool.get_scheduler();

    std::array<std::thread::id, 2> bulk_thr;
    std::thread::id thr_A, thr_B;

    auto start = ::stdexec::schedule(scheduler)
               | ::stdexec::bulk(
                     ::stdexec::par,
                     2,
                     [&](const auto index) -> void { bulk_thr.at(index) = std::this_thread::get_id(); })
               | ::exec::split();

    auto node_A = start | ::stdexec::then([&] { thr_A = std::this_thread::get_id(); });
    auto node_B = std::move(start) | ::stdexec::then([&] { thr_B = std::this_thread::get_id(); });

    auto chain = ::stdexec::when_all(std::move(node_A), std::move(node_B));

    ::stdexec::sync_wait(std::move(chain));

    //! Each @c bulk work item has been processed by a different thread.
    ASSERT_NE(bulk_thr.at(0), bulk_thr.at(1));

    //! However, both asynchronous @c then have been run by the same thread.
    ASSERT_EQ(thr_A, thr_B);
}

class SplitTest
    : public utils::StaticThreadPool<'A', 'B', 'C'>
    , public ::testing::Test {
   public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
    static constexpr size_t index_of_C = index_of<'C'>();
};

//! @test The sender returned by @c exec::split or @c stdexec::when_all does not have a completion scheduler.
TEST_F(SplitTest, with_transition) {
    ::stdexec::scheduler auto scheduler_A = this->pools.at(index_of_A).get_scheduler();
    ::stdexec::scheduler auto scheduler_B = this->pools.at(index_of_B).get_scheduler();
    ::stdexec::scheduler auto scheduler_C = this->pools.at(index_of_C).get_scheduler();

    std::array<std::thread::id, 9> thrids;

    //! The starting chain has a completion scheduler already.
    ::stdexec::sender auto start = ::stdexec::schedule(scheduler_A) | THEN_STORE_ID(thrids.at(0));

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(start), ::stdexec::env<>>,
                  ::exec::_pool_::_static_thread_pool::domain
    >);
    static_assert(has_completion_scheduler_for<decltype(start), ::stdexec::set_value_t>);

    //! However, once @c exec::split is used, there is no completion scheduler anymore.
    ::stdexec::sender auto fork_one = std::move(start) | ::exec::split(); // NOLINT(performance-move-const-arg)

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(fork_one)>,
                  ::stdexec::indeterminate_domain<>
    >);
    static_assert(!has_completion_scheduler_for<decltype(fork_one), ::stdexec::set_value_t>);

    auto stage_one_branch_a = fork_one | THEN_STORE_ID(thrids.at(1));
    auto stage_one_branch_b = std::move(fork_one) | THEN_STORE_ID(thrids.at(2));
    auto stage_one = ::stdexec::when_all(std::move(stage_one_branch_a), std::move(stage_one_branch_b));

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(stage_one)>,
                  ::stdexec::indeterminate_domain<>
    >);
    static_assert(!has_completion_scheduler_for<decltype(stage_one), ::stdexec::set_value_t>);

    auto post_one = std::move(stage_one) | THEN_STORE_ID(thrids.at(3));

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(post_one), ::stdexec::env<>>,
                  ::stdexec::default_domain
    >);
    static_assert(!has_completion_scheduler_for<decltype(post_one), ::stdexec::set_value_t>);

    auto fork_two = std::move(post_one) | ::stdexec::continues_on(scheduler_B) | THEN_STORE_ID(thrids.at(4))
                  | ::exec::split();

    auto stage_two_branch_a = fork_two | ::stdexec::continues_on(scheduler_C) | THEN_STORE_ID(thrids.at(5));
    auto stage_two_branch_b = fork_two | ::stdexec::continues_on(scheduler_A) | THEN_STORE_ID(thrids.at(6));
    auto stage_two_branch_c = std::move(fork_two) | THEN_STORE_ID(thrids.at(7));
    auto stage_two = ::stdexec::when_all(
        std::move(stage_two_branch_a), std::move(stage_two_branch_b), std::move(stage_two_branch_c));

    //! The execution context of the trailing sender is unknown.
    auto post_two = std::move(stage_two) | THEN_STORE_ID(thrids.at(8));

    ::stdexec::sync_wait(std::move(post_two));

    std::cout << "> A: " << threads.at(index_of_A) << std::endl;
    std::cout << "> B: " << threads.at(index_of_B) << std::endl;
    std::cout << "> C: " << threads.at(index_of_C) << std::endl;
    std::cout << ">  : " << std::this_thread::get_id() << std::endl;

    ASSERT_THAT(
        thrids,
        ::testing::ElementsAre(
            threads.at(index_of_A),
            ::testing::AnyOf(
                threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_C), std::this_thread::get_id()),
            ::testing::AnyOf(
                threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_C), std::this_thread::get_id()),
            ::testing::AnyOf(
                threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_C), std::this_thread::get_id()),
            threads.at(index_of_B),
            threads.at(index_of_C),
            threads.at(index_of_A),
            ::testing::AnyOf(
                threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_C), std::this_thread::get_id()),
            ::testing::AnyOf(
                threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_C), std::this_thread::get_id())));
}

} // namespace tests::stdexec::adaptors
