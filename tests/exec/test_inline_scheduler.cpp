#include "gtest/gtest.h"

#include "exec/inline_scheduler.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::inline_scheduler
 * -----------------------------------
 *
 * This group of tests check the behavior of @c exec::inline_scheduler.
 *
 * The tests can be found in @ref exec/test_inline_scheduler.cpp.
 */

namespace tests::exec
{

//! @test Check that the work is scheduled on the main thread.
TEST(inline_scheduler, main_thread)
{
    const auto main = std::this_thread::get_id();

    auto chain = ::stdexec::schedule(::exec::inline_scheduler{})
            | ::stdexec::then([] () -> std::thread::id { return std::this_thread::get_id(); });

    const auto [who] = ::stdexec::sync_wait(std::move(chain)).value(); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(who, main);
}

} // namespace tests::exec
