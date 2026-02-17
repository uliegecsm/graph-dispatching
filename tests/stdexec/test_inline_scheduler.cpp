#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::inline_scheduler
 * ---------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::inline_scheduler.
 *
 * The tests can be found in @ref stdexec/test_inline_scheduler.cpp.
 */

namespace tests::stdexec {

//! @test Check the forward progress guarantee of @c stdexec::inline_scheduler.
TEST(inline_scheduler, forward_progress_guarantee) {
    static_assert(
        ::stdexec::get_forward_progress_guarantee(::stdexec::inline_scheduler{})
        == ::stdexec::forward_progress_guarantee::weakly_parallel);
}

/**
 * @test Check that the work is scheduled on the main thread.
 *
 * This test also checks that the inline scheduler does not execute work before
 * calling @c stdexec::sync_wait.
 */
TEST(inline_scheduler, main_thread) {
    const auto main = std::this_thread::get_id();

    const auto before_chain = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point executed;

    auto chain = ::stdexec::schedule(::stdexec::inline_scheduler{}) | ::stdexec::then([&]() -> std::thread::id {
                     executed = std::chrono::steady_clock::now();
                     return std::this_thread::get_id();
                 });

    const auto before_submit = std::chrono::steady_clock::now();

    const auto [who] = ::stdexec::sync_wait(std::move(chain)).value(); // NOLINT(performance-move-const-arg)

    const auto after_submit = std::chrono::steady_clock::now();

    ASSERT_EQ(who, main);

    ASSERT_LT(before_chain, executed);
    ASSERT_LT(before_submit, executed);
    ASSERT_GT(after_submit, executed);
}

} // namespace tests::stdexec
