#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
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

class BulkTest : public utils::StaticThreadPool<'A'>, public ::testing::Test {};

//! @test Simple @c stdexec::bulk that uses a functor.
TEST_F(BulkTest, bulk_with_functor)
{
    constexpr size_t size = 4;

    ::stdexec::scheduler auto scheduler = pools.front().get_scheduler();

    /// The kernel will execute @ref utils::FillWithThreadID.
    ::utils::FillWithThreadID functor {.ids = std::make_shared<std::vector<size_t>>(size, 0)};

    ::stdexec::sender auto chain = ::stdexec::schedule(scheduler) | ::stdexec::bulk(size, functor);

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_THAT(*functor.ids.get(), ::testing::Each(std::hash<std::thread::id>{}(threads.front())));
}

} // namespace tests::stdexec::adaptors
