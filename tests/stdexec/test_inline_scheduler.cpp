#include "gtest/gtest.h"

#include "stdexec/execution.hpp"

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

namespace tests::stdexec
{

//! @test Check the forward progress guarantee of @c stdexec::inline_scheduler.
TEST(inline_scheduler, forward_progress_guarantee)
{
    static_assert(::stdexec::get_forward_progress_guarantee(::stdexec::inline_scheduler{})
      == ::stdexec::forward_progress_guarantee::weakly_parallel);
}

//! @test Check that the work is scheduled on the main thread.
TEST(inline_scheduler, main_thread)
{
    const auto main = std::this_thread::get_id();

    auto chain = ::stdexec::schedule(::stdexec::inline_scheduler{})
            | ::stdexec::then([] () -> std::thread::id { return std::this_thread::get_id(); });

    const auto [who] = ::stdexec::sync_wait(std::move(chain)).value(); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(who, main);
}

} // namespace tests::stdexec
