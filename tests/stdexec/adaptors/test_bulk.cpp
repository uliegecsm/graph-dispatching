#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED_DEPRECATED_ATTRIBUTES
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

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

namespace tests::stdexec::adaptors {

class BulkTest
    : public utils::StaticThreadPool<'A'>
    , public ::testing::Test { };

//! @test Simple @c stdexec::bulk that uses a functor.
TEST_F(BulkTest, bulk_with_functor) {
    constexpr size_t size = 4;

    /// The kernel will execute @ref utils::FillWithThreadID.
    const ::utils::FillWithThreadID functor{.ids = std::make_shared<std::vector<size_t>>(size, 0)};

    auto work = ::stdexec::just() | ::stdexec::bulk(::stdexec::par, size, functor);

    using work_t = decltype(work);

    ::stdexec::sender auto chain = ::stdexec::starts_on(pools.front().get_scheduler(), std::move(work));

    ::stdexec::sync_wait(std::move(chain));

    ASSERT_THAT(*functor.ids.get(), ::testing::Each(std::hash<std::thread::id>{}(threads.front())));

    //! Let's perform some traits checks on the chain.
    using chain_t = decltype(chain);

    //! The chain environment advertises the default domain, and completes on the @c exec::static_thread_pool domain.
    static_assert(std::same_as<::stdexec::__domain_of_t<::stdexec::env_of_t<chain_t>>, ::stdexec::default_domain>);
    static_assert(std::same_as<
                  ::stdexec::__detail::__completing_domain_t<::stdexec::set_value_t, chain_t>,
                  ::exec::_pool_::_static_thread_pool::domain
    >);

    //! It has a completion scheduler for the value channel.
    static_assert(std::same_as<
                  decltype(::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(chain))),
                  ::exec::_pool_::_static_thread_pool::scheduler
    >);

    /// However, @c work does not know where it completes until it knows where it is started.
    /// See also @cite P3826R3.
    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, work_t>,
                  ::stdexec::indeterminate_domain<>
    >);

    const auto env = ::stdexec::env(::stdexec::prop{::stdexec::get_scheduler, pools.front().get_scheduler()});

    static_assert(std::same_as<
                  ::stdexec::__completion_domain_of_t<::stdexec::set_value_t, decltype(work), decltype(env)>,
                  ::exec::_pool_::_static_thread_pool::domain
    >);
}

} // namespace tests::stdexec::adaptors
