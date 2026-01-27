#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
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

namespace tests::exec {

class StaticThreadPoolTest
    : public utils::StaticThreadPool<'A', 'B', 'C'>
    , public ::testing::Test {
   public:
    static constexpr size_t index_of_A = index_of<'A'>();
    static constexpr size_t index_of_B = index_of<'B'>();
    static constexpr size_t index_of_C = index_of<'C'>();
};

//! @test Check the forward progress guarantee of the @c exec::static_thread_pool scheduler.
TEST_F(StaticThreadPoolTest, forward_progress_guarantee) {
    const auto fpg = ::stdexec::get_forward_progress_guarantee(::exec::static_thread_pool{}.get_scheduler());
    ASSERT_EQ(fpg, ::stdexec::forward_progress_guarantee::parallel);
}

/**
 * @test Check that @c then reports its completion domain when given a starting environment.
 *
 * Inspired by @cite P3826R3.
 */
TEST_F(StaticThreadPoolTest, then_completion_domain_with_env) {
    //! @c work doesn't know where it will complete until it knows where it will start.
    auto work = ::stdexec::just() | THEN_RETURN_ID;

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(work)>,
                  stdexec::indeterminate_domain<>
    >);

    /// By passing the thread pool scheduler environment to @c work,
    /// it knows where it will start, and @c work then correctly reports it will complete on that thread pool's domain.
    const auto env = ::stdexec::env(::stdexec::prop{::stdexec::get_scheduler, pools.front().get_scheduler()});

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(work), decltype(env)>,
                  ::exec::_pool_::_static_thread_pool::domain
    >);

    ASSERT_EQ(
        std::get<0>(::stdexec::sync_wait(
                        ::stdexec::starts_on(
                            pools.front().get_scheduler(), std::move(work)) // NOLINT(performance-move-const-arg)
                        )
                        .value()),
        threads.front());
}

//! @test It supports @c on and @c continues_on.
TEST_F(StaticThreadPoolTest, transitioning) {
    std::array<std::thread::id, 4> thrids;

    auto work = ::stdexec::schedule(pools.at(index_of_A).get_scheduler()) | THEN_STORE_ID(thrids[0])
              | ::stdexec::on(pools.at(index_of_B).get_scheduler(), THEN_STORE_ID(thrids[1]))
              | ::stdexec::continues_on(pools.at(index_of_C).get_scheduler()) | THEN_STORE_ID(thrids[2])
              | THEN_STORE_ID(thrids[3]);

    ::stdexec::sync_wait(std::move(work)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        thrids,
        ::testing::ElementsAre(
            threads.at(index_of_A), threads.at(index_of_B), threads.at(index_of_C), threads.at(index_of_C)));
}

} // namespace tests::exec
