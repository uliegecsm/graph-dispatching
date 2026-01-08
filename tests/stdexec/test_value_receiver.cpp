#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c tests::stdexec::value_receiver
 * -------------------------------------------
 *
 * This group of tests check the behavior of @ref tests::stdexec::value_receiver.
 *
 * The test can be found in @ref tests/stdexec/test_value_receiver.cpp.
 */

namespace tests::stdexec
{

//! @test Check that @ref tests::stdexec::value_receiver works as expected for @c int.
TEST(value_receiver, int)
{
    ::stdexec::sender auto sndr = ::stdexec::just(int{42}) | ::stdexec::continues_on(::stdexec::inline_scheduler{});

    int placeholder = 0;

    auto opstate = ::stdexec::connect(sndr, value_receiver<int>{.value = std::addressof(placeholder)});

    ::stdexec::start(opstate);

    ASSERT_EQ(placeholder, 42);
}

} // namespace tests::stdexec
