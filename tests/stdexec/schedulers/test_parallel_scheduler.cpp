#include "gtest/gtest.h"

#define STDEXEC_SYSTEM_CONTEXT_HEADER_ONLY

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED_DEPRECATED_ATTRIBUTES
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::parallel_scheduler
 * ----------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::parallel_scheduler.
 *
 * It's based on @cite P2079R10.
 *
 * The tests can be found in @ref tests/stdexec/schedulers/test_parallel_scheduler.cpp.
 */

namespace tests::stdexec::schedulers {

struct ParallelSchedulerTest : public ::testing::Test {
    ::stdexec::parallel_scheduler schd = ::stdexec::get_parallel_scheduler();
};

//! @test Check the forward progress guarantee of the @c stdexec::parallel_scheduler scheduler.
TEST_F(ParallelSchedulerTest, forward_progress_guarantee) {
    const auto fpg = ::stdexec::get_forward_progress_guarantee(schd);
    ASSERT_EQ(fpg, ::stdexec::forward_progress_guarantee::parallel);
}

//! @test Check who's executing the work.
TEST_F(ParallelSchedulerTest, who) {
    const auto main = std::this_thread::get_id();

    auto chain = ::stdexec::schedule(schd)
               | ::stdexec::then([]() -> std::thread::id { return std::this_thread::get_id(); });

    const auto [worker] = ::stdexec::sync_wait(std::move(chain)).value();

    ASSERT_NE(main, worker);
}

} // namespace tests::stdexec::schedulers
