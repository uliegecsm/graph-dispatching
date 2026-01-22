#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/fork_join.hpp"
#include "exec/repeat_effect_until.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/Counter.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::fork_join
 * ----------------------------
 *
 * This group of tests check the behavior of @c exec::fork_join.
 *
 * The tests can be found in @ref tests/exec/adaptors/test_fork_join.cpp.
 */

#if defined(__clang__) && defined(__HIPCC__) && (__clang_major__ == 19 && __clang_minor__ == 0 && __clang_patch__ == 0)
#    define SKIP_FOR_HIPCC_19 GTEST_SKIP() << "This test is known to seg. fault if compiled with hipcc 19.0.0.";
#else
#    define SKIP_FOR_HIPCC_19
#endif

namespace tests::exec::adaptors {

class ForkJoinTest
    : public ::utils::StaticThreadPool<'A'>
    , public ::testing::Test {
   public:
    static constexpr size_t index_of_A = index_of<'A'>();
};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CHAIN(_schd_)                                                                                                  \
    ::stdexec::schedule(_schd_) | ::stdexec::then([&witness]() noexcept -> int {                                       \
        ++witness;                                                                                                     \
        return witness;                                                                                                \
    })                                                                                                                 \
        | ::exec::fork_join(                                                                                           \
            ::tests::stdexec::check_scheduler<::tests::stdexec::default_scheduler_t>()                                 \
                | ::stdexec::then([&witness](const int received) noexcept { witness += received; }),                   \
            ::tests::stdexec::check_scheduler<::tests::stdexec::default_scheduler_t>()                                 \
                | ::stdexec::then([&witness](const int received) noexcept { witness += received; })                    \
                | ::stdexec::then(::tests::utils::Counter{}))

/**
 * @test Check that @c exec::fork_join effectively forks (and passes the values along), and joins.
 *
 * It mimics what could be achieved with @c stdexec::split and @c stdexec::when_all.
 *
 * @warning According to @cite P3682R0, @c stdexec::split will be removed from the proposed @c std::execution for C++26.
 */
TEST_F(ForkJoinTest, copies) {
    SKIP_FOR_HIPCC_19
    ::tests::utils::Counter::reset();

    std::atomic<int> witness{0};

    auto chain = CHAIN(pools.at(index_of_A).get_scheduler());

    using chain_t = decltype(chain);

    static_assert(std::same_as<
                  std::invoke_result_t<::stdexec::get_completion_signatures_t, chain_t>,
                  ::stdexec::completion_signatures<::stdexec::set_stopped_t(), ::stdexec::set_value_t()>
    >);

    /// The completion domain after the @c exec::fork_join is the one of the upstream (before the fork). Compared to @c stdexec::when_all
    /// that does not know about the sender that lives *before* the fork, @c exec::fork_join has that knowledge.
    static_assert(std::same_as<::stdexec::__domain_of_t<::stdexec::env_of_t<chain_t>>, ::stdexec::default_domain>);
    static_assert(std::same_as<
                  ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, chain_t>,
                  ::exec::_pool_::_static_thread_pool::domain
    >);
    static_assert(std::same_as<
                  decltype(::stdexec::get_completion_domain<::stdexec::set_value_t>(
                      ::stdexec::get_env(chain), ::stdexec::env<>{})),
                  ::exec::_pool_::_static_thread_pool::domain
    >);
    static_assert(std::same_as<
                  decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(
                      ::stdexec::get_env(chain), ::stdexec::env<>{})),
                  ::exec::_pool_::_static_thread_pool::scheduler
    >);

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(witness, 3);
    ASSERT_EQ(::tests::utils::Counter::copy_constructions.load(), 0);
}

/**
 * @test Using @c exec::fork_join within a @c exec::repeat_effect_until does not memoize results.
 *
 * As opposed to @c stdexec::split that makes a single execution and memoizes the result, it seems that @c exec::fork_join
 * does not memoize the result in between the repetitions. But it leads to copying the input sender at each iteration.
 *
 * See also @ref tests::exec::adaptors::RepeatEffectUntilTest_split_Test.
 */
TEST_F(ForkJoinTest, fork_join_does_not_memoize_results) {
    SKIP_FOR_HIPCC_19
    ::tests::utils::Counter::reset();

    unsigned short int irep = 0;
    std::atomic<int> witness = 0;

    auto loop = ::exec::repeat_effect_until(
        CHAIN(pools.at(index_of_A).get_scheduler()) | ::stdexec::then([&irep]() -> bool {
            std::printf(
                "Repetition %u: copy constructed %d times.\n",
                irep++,
                ::tests::utils::Counter::copy_constructions.load());
            return irep >= 5;
        }));

    ::stdexec::sync_wait(std::move(loop));

    ASSERT_EQ(witness, (std::pow(3, (irep - 1) + 2) - 3) / 2);
    ASSERT_EQ(::tests::utils::Counter::copy_constructions.load(), 5);
}

} // namespace tests::exec::adaptors
