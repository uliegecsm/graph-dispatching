#include <vector>

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
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::sync_wait
 * -------------------------------
 *
 * This group of tests check the behavior of @c stdexec::sync_wait.
 *
 * The tests can be found in @ref tests/stdexec/adaptors/test_sync_wait.cpp.
 */

namespace tests::stdexec::adaptors {

class SyncWaitTest
    : public utils::StaticThreadPool<'A'>
    , public ::testing::Test { };

/// @test Simple @c stdexec::sync_wait test that blocks the current thread until the result is known.
TEST_F(SyncWaitTest, block) {
    ::stdexec::sender auto chain = ::stdexec::schedule(std::get<0>(this->pools).get_scheduler())
                                 | ::stdexec::then([this]() -> double {
                                       if (std::this_thread::get_id() == this->main)
                                           throw std::runtime_error("Unexpected.");
                                       return 42.;
                                   });

    const auto [result] = ::stdexec::sync_wait(std::move(chain)).value(); // NOLINT(performance-move-const-arg)

    static_assert(std::same_as<decltype(result), const double>);

    ASSERT_EQ(result, 42.);
}

} // namespace tests::stdexec::adaptors
