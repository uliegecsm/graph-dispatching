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

#include "tests/Utils.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests related to the @c stdexec::run_loop scheduler
 * ---------------------------------------------------
 *
 * This group of tests check behavior related to the @c stdexec::run_loop scheduler.
 *
 * The tests can be found in @ref tests/stdexec/schedulers/test_run_loop.cpp.
 */

namespace tests::stdexec::schedulers {

//! The @c stdexec::run_loop uses the thread that calls its @c run method to execute work.
TEST(RunLoopTest, thread) {
    const auto main = std::this_thread::get_id();

    ::stdexec::run_loop loop{};

    std::thread::id worker;

    auto sndr = ::stdexec::schedule(loop.get_scheduler())
              | ::stdexec::then([]() noexcept -> std::thread::id { return std::this_thread::get_id(); });

    auto opstate = ::stdexec::connect(
        std::move(sndr), ::tests::stdexec::ValueReceiver{std::addressof(worker)}); // NOLINT(performance-move-const-arg)

    ::stdexec::start(opstate);

    loop.finish();
    loop.run();

    ASSERT_EQ(worker, main);
}

} // namespace tests::stdexec::schedulers
