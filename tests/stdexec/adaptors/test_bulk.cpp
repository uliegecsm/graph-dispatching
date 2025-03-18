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
    const ::utils::FillWithThreadID functor {.ids = std::make_shared<std::vector<size_t>>(size, 0)};

    ::stdexec::sender auto chain = ::stdexec::schedule(scheduler) | ::stdexec::bulk(size, functor);

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_THAT(*functor.ids.get(), ::testing::Each(std::hash<std::thread::id>{}(threads.front())));

    //! Let's perform some traits checks on the chain.
    using chain_t = decltype(chain);

    //! The chain's environment cannot itself be queried for its domain.
    static_assert(!::stdexec::tag_invocable<::stdexec::get_domain_t, ::stdexec::env_of_t<chain_t>>);

    //! However, it has a completion scheduler for the value channel.
    static_assert(::stdexec::__has_completion_scheduler<chain_t, ::stdexec::set_value_t>);

    static_assert(std::same_as<
        ::stdexec::__detail::__completion_scheduler_for<::stdexec::env_of_t<chain_t>, ::stdexec::set_value_t>,
        exec::_pool_::static_thread_pool_::scheduler
    >);

    //! Therefore, it has a **non-default early** completion domain.
    static_assert(std::same_as<
        ::stdexec::__detail::__completion_domain_of<chain_t>,
        exec::_pool_::static_thread_pool_::domain
    >);

    static_assert(std::same_as<::stdexec::__early_domain_of_t<chain_t>, exec::_pool_::static_thread_pool_::domain>);
}

} // namespace tests::stdexec::adaptors
