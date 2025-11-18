#include <vector>

#include "gmock/gmock.h"

#include "hpx/execution.hpp"
#include "hpx/runtime.hpp"
#include "hpx/thread.hpp"

#include "tests/Utils.hpp"
#include "tests/hpx/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c hpx::execution::experimental::bulk
 * -----------------------------------------------
 *
 * This group of tests check the behavior of @c hpx::execution::experimental::bulk.
 *
 * The test can be found in @ref hpx/adaptors/test_bulk.cpp.
 */

namespace tests::hpx::adaptors
{

using ::utils::hpx::test::HPXTest;

//! @test Simple @c hpx::execution::experimental::bulk that uses a functor.
TEST_F(HPXTest, bulk_with_functor)
{
    namespace execution = ::hpx::execution::experimental;

    constexpr size_t size = 4;

    //! Create a thread pool with a single thread.
    ASSERT_TRUE(::hpx::resource::pool_exists(this->hpx->single_thread_pool_name));
    auto& pool = ::hpx::resource::get_thread_pool(this->hpx->single_thread_pool_name);
    ASSERT_EQ(pool.get_os_thread_count(), 1);

    //! Get a scheduler for that thread pool with asynchronous execution.
    const execution::thread_pool_scheduler scheduler(&pool, ::hpx::launch::async);

    //! Retrieve the thread ID.
    const auto [pool_thread_ID] = ::hpx::this_thread::experimental::sync_wait(
        execution::schedule(scheduler) | execution::then([]() -> size_t { return ::utils::get_thread_id(); } )).value();

    ASSERT_NE(pool_thread_ID, 0);

    /// The kernel will execute @ref utils::FillWithThreadID.
    const ::utils::FillWithThreadID functor {.ids = std::make_shared<std::vector<size_t>>(size, 0)};

    auto chain = execution::schedule(scheduler) | execution::bulk(size, functor);

    ::hpx::this_thread::experimental::sync_wait(std::move(chain));

    ASSERT_THAT(*functor.ids.get(), ::testing::Each(pool_thread_ID));
}

} // namespace tests::hpx::adaptors
