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

} // namespace tests::stdexec::adaptors
