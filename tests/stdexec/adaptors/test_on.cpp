#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/on.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::on
 * ---------------------
 *
 * This group of tests check the behavior of @c exec::on.
 *
 * The test can be found in @ref stdexec/adaptors/test_on.cpp.
 */

namespace tests::stdexec::adaptors
{

//! @test Simple test for @c exec::on.
TEST(exec, on)
{
    exec::static_thread_pool pool_A{1}, pool_B{1};

    ::stdexec::scheduler auto scheduler_A = pool_A.get_scheduler();
    ::stdexec::scheduler auto scheduler_B = pool_B.get_scheduler();

    std::thread::id thr_A, thr_B;
    size_t counter = 0;

    auto chain = ::stdexec::schedule(scheduler_A)
        |                         ::stdexec::then([&]() -> void { thr_A = std::this_thread::get_id(); ++counter; })
        | ::exec::on(scheduler_B, ::stdexec::then([&]() -> void { thr_B = std::this_thread::get_id(); ++counter; }));

    ::stdexec::sync_wait(std::move(chain));

    //! The second @c then has indeed been executed by the second thread pool.
    ASSERT_NE(thr_A, thr_B);

    ASSERT_EQ(counter, 2);
}

} // namespace tests::stdexec::adaptors
