#include <vector>

#include "gmock/gmock.h"
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
 * Tests for @c stdexec::bulk
 * --------------------------
 *
 * This group of tests check the behavior of @c stdexec::bulk.
 *
 * The test can be found in @ref stdexec/adaptors/test_bulk.cpp.
 */

namespace tests::stdexec::adaptors
{

//! @test Simple @c stdexec::bulk that uses a functor.
TEST(stdexec, bulk_with_functor)
{
    constexpr size_t size = 4;

    //! Create a thread pool with a single thread.
    exec::static_thread_pool pool{1};

    ::stdexec::scheduler auto scheduler = pool.get_scheduler();

    const auto [pool_thread_ID] = ::stdexec::sync_wait(
        ::stdexec::schedule(scheduler) | ::stdexec::then([]() -> size_t { return ::utils::get_thread_id(); } )).value();

    ASSERT_NE(pool_thread_ID, 0);

    /// The kernel will execute @ref utils::FillWithThreadID.
    ::utils::FillWithThreadID functor {.ids = std::make_shared<std::vector<size_t>>(size, 0)};

    ::stdexec::sender auto chain = ::stdexec::schedule(scheduler) | ::stdexec::bulk(size, functor);

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_THAT(*functor.ids.get(), ::testing::Each(pool_thread_ID));
}

} // namespace tests::stdexec::adaptors
