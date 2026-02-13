#include <chrono>
#include <format>
#include <iostream>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "exec/ensure_started.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::ensure_started
 * ------------------------------------
 *
 * This group of tests check the behavior of @c exec::ensure_started.
 *
 * The test can be found in @ref tests/exec/adaptors/test_ensure_started.cpp.
 *
 * @note In the long term, @c exec::ensure_started might be dropped for @c exec::async_scope.
 */

namespace tests::stdexec::adaptors {

class EnsureStartedTest
    : public utils::StaticThreadPool<'A'>
    , public ::testing::Test { };

/// @test Simple @c exec::ensure_started test to check that the senders consumer starts to asynchronously consume the chain.
TEST_F(EnsureStartedTest, asynchronously_consumes) {
    using duration_t = std::chrono::steady_clock::duration;

    const auto start = std::chrono::steady_clock::now();

    ::stdexec::sender auto chain_A = ::stdexec::schedule(std::get<0>(this->pools).get_scheduler())
                                   | ::stdexec::then(
                                         [&start]() -> duration_t { return std::chrono::steady_clock::now() - start; });

    const auto before_ensure_started = std::chrono::steady_clock::now();

    //! Ensure that we start consuming.
    auto handle_A = ::exec::ensure_started(chain_A);

    //! We continue adding stuff.
    ::stdexec::sender auto chain_B = std::move(chain_A) // NOLINT(performance-move-const-arg)
                                   | ::stdexec::then(
                                         [&start](const auto&) { return std::chrono::steady_clock::now() - start; });

    //! But before we wait for the whole chain, let's explicitly wait for the first part.
    const auto [value_A] = ::stdexec::sync_wait(std::move(handle_A)).value();

    const auto after_sync_wait_A = std::chrono::steady_clock::now();

    const auto [value_B] = ::stdexec::sync_wait(std::move(chain_B)).value(); // NOLINT(performance-move-const-arg)

    const auto end = std::chrono::steady_clock::now();

    //! The work in the first part cannot have started before the call to @c exec::ensure_started.
    ASSERT_LT((before_ensure_started - start).count(), value_A.count());

    //! The work in the first part is finished once we've synchronized its handle.
    ASSERT_LT(value_A.count(), (after_sync_wait_A - start).count());

    //! Then only comes the work of the second part.
    ASSERT_LT((after_sync_wait_A - start).count(), value_B.count());

    ASSERT_LT(value_B.count(), (end - start).count());
}

} // namespace tests::stdexec::adaptors
