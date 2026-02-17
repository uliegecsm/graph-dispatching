#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wmissing-field-initializers")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::read_env
 * ------------------------------
 *
 * This group of tests check the behavior of @c stdexec::read_env.
 *
 * The tests can be found in @ref tests/stdexec/factories/test_read_env.cpp.
 */

namespace tests::stdexec::factories {

//! @test Use @c stdexec::read_env to retrieve the default scheduler type.
TEST(read_env, get_scheduler_default) {
    ::stdexec::sync_wait(
        ::stdexec::read_env(::stdexec::get_scheduler) | ::stdexec::then([](const auto& scheduler) {
            static_assert(
                std::same_as<std::remove_cvref_t<decltype(scheduler)>, ::tests::stdexec::default_scheduler_t>);
        })
        | THEN_SHOW_ID);
}

//! @test Use @c stdexec::read_env to retrieve the static thread pool scheduler type.
TEST(read_env, get_scheduler_static_thread_pool) {
    ::exec::static_thread_pool pool{1};

    ::stdexec::sync_wait(::stdexec::schedule(pool.get_scheduler()) | ::stdexec::let_value([] {
                             return ::stdexec::read_env(::stdexec::get_scheduler)
                                  | ::stdexec::then([](const auto& scheduler) {
                                        static_assert(std::same_as<
                                                      std::remove_cvref_t<decltype(scheduler)>,
                                                      ::exec::_pool_::_static_thread_pool::scheduler
                                        >);
                                    })
                                  | THEN_SHOW_ID;
                         }));
}

} // namespace tests::stdexec::factories
