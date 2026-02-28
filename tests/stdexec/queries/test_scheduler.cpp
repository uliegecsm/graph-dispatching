#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wmissing-field-initializers")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests related to the scheduler query
 * ------------------------------------
 *
 * This group of tests check behavior related to the scheduler query.
 *
 * The tests can be found in @ref tests/stdexec/queries/test_scheduler.cpp.
 */

namespace tests::stdexec::queries {

/**
 * @test @c stdexec::get_scheduler returns a sender that will read the environment
 *       for the scheduler query.
 */
TEST(Scheduler, get_scheduler) {
    auto get_schd = ::stdexec::get_scheduler();
    static_assert(std::same_as<
                  ::stdexec::__demangle_t<decltype(get_schd)>,
                  ::tests::stdexec::basic_sender<::stdexec::__read_env_t, ::stdexec::get_scheduler_t>
    >);

    auto [res] = ::stdexec::sync_wait(::stdexec::on(::stdexec::inline_scheduler{}, std::move(get_schd))).value();

    static_assert(std::same_as<decltype(res), ::stdexec::run_loop::scheduler>);
}

} // namespace tests::stdexec::queries
