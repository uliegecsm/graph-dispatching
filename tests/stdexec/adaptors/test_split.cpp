#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::split
 * ---------------------------
 *
 * This group of tests check the behavior of @c stdexec::split.
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
        | ::stdexec::bulk(2, [&](const auto index) -> void { bulk_thr[index] = std::this_thread::get_id(); })
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

} // namespace tests::stdexec::adaptors
