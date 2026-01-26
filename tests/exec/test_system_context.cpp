#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "exec/system_context.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::parallel_scheduler
 * -------------------------------------
 *
 * This group of tests check the behavior of @c exec::parallel_scheduler.
 *
 * It's based on @cite P2079R10.
 *
 * The tests can be found in @ref exec/test_system_context.cpp.
 */

namespace tests::exec {

struct SystemContextTest : public ::testing::Test {
    ::exec::parallel_scheduler schd = ::exec::get_parallel_scheduler();
};

//! @test Check the forward progress guarantee of the @c exec::system_context scheduler.
TEST_F(SystemContextTest, forward_progress_guarantee) {
    const auto fpg = ::stdexec::get_forward_progress_guarantee(schd);
    ASSERT_EQ(fpg, ::stdexec::forward_progress_guarantee::parallel);
}

//! @test Check who's executing the work.
TEST_F(SystemContextTest, who) {
    const auto main = std::this_thread::get_id();

    auto chain = ::stdexec::schedule(schd)
               | ::stdexec::then([]() -> std::thread::id { return std::this_thread::get_id(); });

    const auto [worker] = ::stdexec::sync_wait(std::move(chain)).value();

    ASSERT_NE(main, worker);
}

} // namespace tests::exec
