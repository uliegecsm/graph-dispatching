#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED_DEPRECATED_ATTRIBUTES
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c tests::stdexec::ValueReceiver
 * ------------------------------------------
 *
 * This group of tests check the behavior of @ref tests::stdexec::ValueReceiver.
 *
 * The tests can be found in @ref tests/stdexec/test_value_receiver.cpp.
 */

namespace tests::stdexec {

//! @test Check that @ref tests::stdexec::ValueReceiver works as expected for @c int.
TEST(ValueReceiver, int) {
    ::stdexec::sender auto sndr = ::stdexec::just(int{42}) | ::stdexec::continues_on(::stdexec::inline_scheduler{});

    int placeholder = 0; // NOLINT(misc-const-correctness)

    auto opstate = ::stdexec::connect(sndr, ValueReceiver<int>{.value = std::addressof(placeholder)});

    ::stdexec::start(opstate);

    ASSERT_EQ(placeholder, 42);
}

} // namespace tests::stdexec
