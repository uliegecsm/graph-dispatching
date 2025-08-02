#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::static_thread_pool
 * -------------------------------------
 *
 * This group of tests check the behavior of @c exec::static_thread_pool.
 *
 * The tests can be found in @ref exec/test_static_thread_pool.cpp.
 */

namespace tests::exec
{

class StaticThreadPoolTest : public utils::StaticThreadPool<'A', 'B', 'C'>, public ::testing::Test
{
public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
    static constexpr size_t index_of_C = index_of<'C'>();
};

//! @test Check that @c then has an early domain.
TEST_F(StaticThreadPoolTest, then_early_domain)
{
    auto work = ::stdexec::schedule(pools.front().get_scheduler()) | THEN_RETURN_ID;

    static_assert(std::same_as<::stdexec::__early_domain_of_t<decltype(work)>, ::exec::_pool_::static_thread_pool_::domain>);

    ASSERT_EQ(std::get<0>(::stdexec::sync_wait(std::move(work)).value()), threads.front()); // NOLINT(performance-move-const-arg)
}

//! @test It supports @c on and @c continues_on.
TEST_F(StaticThreadPoolTest, transitioning)
{
    std::array<std::thread::id, 4> thrids;

    auto work = ::stdexec::schedule(pools.at(index_of_A).get_scheduler())
        | THEN_STORE_ID(thrids[0])
        | ::stdexec::on(pools.at(index_of_B).get_scheduler(), THEN_STORE_ID(thrids[1]))
        | ::stdexec::continues_on(pools.at(index_of_C).get_scheduler())
        | THEN_STORE_ID(thrids[2])
        | THEN_STORE_ID(thrids[3]);

    ::stdexec::sync_wait(std::move(work)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(thrids, ::testing::ElementsAre(
        threads.at(index_of_A),
        threads.at(index_of_B),
        threads.at(index_of_C),
        threads.at(index_of_C)));
}

} // namespace tests::exec
