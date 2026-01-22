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
 * Tests for @c stdexec::when_all
 * ------------------------------
 *
 * This group of tests check the behavior of @c stdexec::when_all.
 *
 * The test can be found in @ref tests/stdexec/adaptors/test_when_all.cpp.
 */

namespace tests::stdexec::adaptors
{

//! @test Same as https://github.com/NVIDIA/stdexec/blob/970dbace4ad52a38c9b18665d077f14159792b23/test/stdexec/algos/adaptors/test_when_all.cpp#L253.
TEST(when_all, propagates_completion_domain_from_same_type_children)
{
    ::exec::static_thread_pool pool_A{1}, pool_B{1};

    ::stdexec::sender auto sndr = ::stdexec::when_all(
        ::stdexec::starts_on(pool_A.get_scheduler(), ::stdexec::just(13)),
        ::stdexec::starts_on(pool_B.get_scheduler(), ::stdexec::just(3.14))
    );

    static_assert(std::same_as<
                  decltype(::stdexec::get_completion_domain<::stdexec::set_value_t>(
                      ::stdexec::get_env(sndr), ::stdexec::env<>{})),
                  ::exec::_pool_::_static_thread_pool::domain
    >);

    static_assert(!tests::stdexec::has_completion_scheduler_for<decltype(sndr), ::stdexec::set_value_t>);
}

/**
 * @test Inspired by https://github.com/NVIDIA/stdexec/blob/970dbace4ad52a38c9b18665d077f14159792b23/test/stdexec/algos/adaptors/test_when_all.cpp#L253.
 *
 * @warning As stated in @cite P2999R3 (Section 2.1.1, "Dispatching via execution domain tags"), @c stdexec::when_all only accepts a set of senders when they all share a common domain.
 */
TEST(when_all, propagates_completion_domain_from_different_type_children)
{
    ::exec::static_thread_pool pool{1};

    ::stdexec::sender auto sndr = ::stdexec::when_all(
        ::stdexec::starts_on(::stdexec::inline_scheduler{}, ::stdexec::just(13)),
        ::stdexec::starts_on(pool.get_scheduler(), ::stdexec::just(3.14))
    );
    auto domain = ::stdexec::get_completion_domain<::stdexec::set_value_t>(::stdexec::get_env(sndr), ::stdexec::env<>{});
    static_assert(std::same_as<decltype(domain), ::stdexec::default_domain>);
}

class WhenAllTest : public utils::StaticThreadPool<'A', 'B', 'C'>, public ::testing::Test {};

//! @test Check on which thread after @c stdexec::when_all it will execute.
TEST_F(WhenAllTest, on_which_thread_after)
{
    std::thread::id thr_A, thr_B, thr_C, thr_X;

    auto when_all = ::stdexec::when_all(
        ::stdexec::schedule(this->pools.at(index_of<'A'>()).get_scheduler()) | ::stdexec::then([&](){ thr_A = std::this_thread::get_id(); }),
        ::stdexec::schedule(this->pools.at(index_of<'B'>()).get_scheduler()) | ::stdexec::then([&](){ thr_B = std::this_thread::get_id(); }),
        ::stdexec::schedule(this->pools.at(index_of<'C'>()).get_scheduler()) | ::stdexec::then([&](){ thr_C = std::this_thread::get_id(); })
    );

    static_assert(std::same_as<
        std::invoke_result_t<::stdexec::get_completion_domain_t<::stdexec::set_value_t>, ::stdexec::env_of_t<decltype(when_all)>>,
        ::exec::_pool_::_static_thread_pool::domain
    >);

    auto chain = std::move(when_all) // NOLINT(performance-move-const-arg)
        | ::tests::stdexec::check_scheduler<::tests::stdexec::default_scheduler_t>()
        | ::stdexec::then([&]{ thr_X = std::this_thread::get_id(); });

    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(thr_A, this->threads.at(index_of<'A'>()));
    ASSERT_EQ(thr_B, this->threads.at(index_of<'B'>()));
    ASSERT_EQ(thr_C, this->threads.at(index_of<'C'>()));

    ASSERT_THAT(thr_X, ::testing::AnyOf(thr_A, thr_B, thr_C));
}

} // namespace tests::stdexec::adaptors
