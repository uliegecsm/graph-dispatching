#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/split.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c tests::stdexec::check_scheduler
 * --------------------------------------------
 *
 * This group of tests check the behavior of @ref tests::stdexec::check_scheduler.
 *
 * The tests can be found in @ref tests/stdexec/test_check_scheduler.cpp.
 */

namespace tests::stdexec {

using static_thread_pool_scheduler_t = ::exec::_pool_::_static_thread_pool::scheduler;

//! @test Default scheduler.
TEST(check_scheduler, default) {
    ::stdexec::sync_wait(::stdexec::just() | check_scheduler<default_scheduler_t>() | THEN_SHOW_ID);
}

//! @test @c exec::static_thread_pool scheduler.
TEST(check_scheduler, static_thread_pool) {
    ::exec::static_thread_pool pool{1};

    auto chain = ::stdexec::schedule(pool.get_scheduler()) | check_scheduler<static_thread_pool_scheduler_t>()
               | THEN_SHOW_ID;

    ::stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
}

/**
 * @test @c stdexec::when_all does not forward the scheduler to its input senders.
 *
 * See also https://github.com/NVIDIA/stdexec/issues/1736#issuecomment-3720622409.
 */
TEST(check_scheduler, split_when_all_no_forward) {
    ::exec::static_thread_pool pool{1};

    auto fork = ::stdexec::schedule(pool.get_scheduler()) | check_scheduler<static_thread_pool_scheduler_t>()
              | THEN_SHOW_ID | ::exec::split();

    auto chain = ::stdexec::when_all(
        fork | THEN_SHOW_ID | check_scheduler<default_scheduler_t>(),
        fork | THEN_SHOW_ID | check_scheduler<default_scheduler_t>());

    ::stdexec::sync_wait(std::move(chain) | check_scheduler<default_scheduler_t>());
}

/**
 * @test @c stdexec::transfer_when_all does not forward the scheduler to its input senders.
 *
 * See also https://github.com/NVIDIA/stdexec/issues/1736#issuecomment-3720622409.
 */
TEST(check_scheduler, split_transfer_when_all_no_forward) {
    ::exec::static_thread_pool pool{1};

    auto fork = ::stdexec::schedule(pool.get_scheduler()) | check_scheduler<static_thread_pool_scheduler_t>()
              | THEN_SHOW_ID | ::exec::split();

    auto chain = ::stdexec::transfer_when_all(
        pool.get_scheduler(),
        fork | THEN_SHOW_ID | check_scheduler<default_scheduler_t>(),
        fork | THEN_SHOW_ID | check_scheduler<default_scheduler_t>());

    ::stdexec::sync_wait(std::move(chain) | check_scheduler<static_thread_pool_scheduler_t>());
}

//! @test Multiple splits.
TEST(check_scheduler, multiple_splits) {
    ::exec::static_thread_pool pool{1};

    auto fork_A = ::stdexec::schedule(pool.get_scheduler()) | check_scheduler<static_thread_pool_scheduler_t>()
                | THEN_SHOW_ID | ::exec::split();

    auto chain_A_branch_a = fork_A | THEN_SHOW_ID;
    auto chain_A_branch_b = std::move(fork_A) | THEN_SHOW_ID;

    auto chain_A = ::stdexec::when_all(std::move(chain_A_branch_a), std::move(chain_A_branch_b)) | THEN_SHOW_ID;

    auto fork_B = std::move(chain_A) | ::stdexec::continues_on(pool.get_scheduler())
                | check_scheduler<static_thread_pool_scheduler_t>() | ::exec::split();

    auto chain_B_branch_a = fork_B | THEN_SHOW_ID | check_scheduler<default_scheduler_t>();
    auto chain_B_branch_b = std::move(fork_B) | THEN_SHOW_ID | check_scheduler<default_scheduler_t>();

    auto chain_B = ::stdexec::when_all(std::move(chain_B_branch_a), std::move(chain_B_branch_b)) | THEN_SHOW_ID
                 | check_scheduler<default_scheduler_t>();

    ::stdexec::sync_wait(std::move(chain_B) | check_scheduler<default_scheduler_t>());
}

} // namespace tests::stdexec
