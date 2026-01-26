#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::on
 * ------------------------
 *
 * This group of tests check the behavior of @c stdexec::on.
 *
 * The test can be found in @ref stdexec/adaptors/test_on.cpp.
 */

namespace tests::stdexec::adaptors {

class OnTest
    : public utils::StaticThreadPool<'A', 'B'>
    , public ::testing::Test {
   public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
};

//! @test Simple test for @c stdexec::on showing that the context transition is indeed temporary.
TEST_F(OnTest, on) {
    ::stdexec::scheduler auto scheduler_A = this->pools.at(index_of_A).get_scheduler();
    ::stdexec::scheduler auto scheduler_B = this->pools.at(index_of_B).get_scheduler();

    std::array<std::thread::id, 3> thrids;
    size_t counter = 0;

    auto chain = ::stdexec::schedule(scheduler_A) | THEN_STORE_ID(thrids[0], { ++counter; })
               | ::stdexec::on(scheduler_B, THEN_STORE_ID(thrids[1], { ++counter; }))
               | THEN_STORE_ID(thrids[2], { ++counter; });

    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    //! The second @c then has indeed been executed by the second thread pool.
    ASSERT_THAT(thrids, ::testing::ElementsAre(threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_A)));

    ASSERT_EQ(counter, 3);
}

//! @test @c stdexec::on can be used nested, and the inner most one wins.
TEST_F(OnTest, on_nested) {
    ::stdexec::scheduler auto scheduler_A = this->pools.at(index_of_A).get_scheduler();
    ::stdexec::scheduler auto scheduler_B = this->pools.at(index_of_B).get_scheduler();

    std::thread::id tid;

    auto chain = ::stdexec::just() | ::stdexec::on(scheduler_A, ::stdexec::on(scheduler_B, THEN_STORE_ID(tid)));
    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, threads.at(index_of_B));
}

} // namespace tests::stdexec::adaptors
