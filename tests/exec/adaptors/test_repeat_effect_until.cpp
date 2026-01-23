#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-result")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "exec/repeat_effect_until.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/utils/Counter.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::repeat_effect_until
 * --------------------------------------
 *
 * This group of tests check the behavior of @c exec::repeat_effect_until.
 *
 * The tests can be found in @ref tests/exec/adaptors/test_repeat_effect_until.cpp.
 */

namespace tests::exec::adaptors {

class RepeatEffectUntilTest
    : public ::utils::StaticThreadPool<'A'>
    , public ::testing::Test {
   public:
    static constexpr size_t index_of_A = index_of<'A'>();
};

#define CHAIN                                                                                                          \
    ::stdexec::schedule(pools.at(index_of_A).get_scheduler()) | ::stdexec::then(::tests::utils::Counter{})             \
        | ::stdexec::then([&witness] { ++witness; })

/**
 * @test Check that the input sender to @c exec::repeat_effect_until (or at least its data) will be copied once per repetition.
 *
 * The @c exec::repeat_effect_until sender stores its input sender. Each re-connection copies the input sender.
 * It is expected from the algorithm implementation.
 */
TEST_F(RepeatEffectUntilTest, copies) {
    ::tests::utils::Counter::reset();

    unsigned short int irep = 0, witness = 0;

    auto chain = ::exec::repeat_effect_until(CHAIN | ::stdexec::then([&irep]() -> bool {
                                                 std::cout << "Repetition " << irep << ": copy constructed "
                                                           << ::tests::utils::Counter::copy_constructions.load()
                                                           << "times.\n";
                                                 return (++irep) >= 5;
                                             }));

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(witness, 5);
    ASSERT_EQ(::tests::utils::Counter::copy_constructions.load(), 5);
}

/**
 * @test Try to mitigate the copies using @c stdexec::split.
 *
 * Since @c stdexec::split makes a single execution and memoizes the result, it effectively avoids the copies but
 * also prevents the work from being repeated. It can be thought of as acting as a cacher for the results.
 */
TEST_F(RepeatEffectUntilTest, split) {
    ::tests::utils::Counter::reset();

    unsigned short int irep = 0, witness = 0;

    auto chain = ::exec::repeat_effect_until(CHAIN | ::stdexec::split() | ::stdexec::then([&irep]() -> bool {
                                                 std::printf( // NOLINT(modernize-use-std-print)
                                                     "Repetition %u: copy constructed %d times.\n",
                                                     irep,
                                                     ::tests::utils::Counter::copy_constructions.load());
                                                 return (++irep) >= 5;
                                             }));

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(witness, 1);
    ASSERT_EQ(::tests::utils::Counter::copy_constructions.load(), 0);
}

} // namespace tests::exec::adaptors
