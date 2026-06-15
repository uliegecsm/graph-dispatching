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
 * Tests for @c tests::stdexec::SinkReceiver
 * -------------------------------------------
 *
 * This group of tests check the behavior of @ref tests::stdexec::SinkReceiver.
 *
 * The tests can be found in @ref tests/stdexec/test_sink_receiver.cpp.
 */

namespace tests::stdexec {

//! @test Check that @ref tests::stdexec::SinkReceiver satisfies the @c ::stdexec::receiver concept.
constexpr bool test_sink_receiver_traits() {
    static_assert(::stdexec::receiver<SinkReceiver>);

    static_assert(
        ::stdexec::receiver_of<SinkReceiver, ::stdexec::completion_signatures<::stdexec::set_value_t(int, double)>>);
    static_assert(::stdexec::receiver_of<
                  SinkReceiver,
                  ::stdexec::completion_signatures<::stdexec::set_error_t(std::exception_ptr)>
    >);
    static_assert(::stdexec::receiver_of<SinkReceiver, ::stdexec::completion_signatures<::stdexec::set_stopped_t()>>);

    return true;
}
static_assert(test_sink_receiver_traits());

//! @test Check that a sender can be connected to @ref tests::stdexec::SinkReceiver.
TEST(SinkReceiver, connect) {
    auto opstate = ::stdexec::connect(::stdexec::just(42), SinkReceiver{});
    ::stdexec::start(opstate);
}

} // namespace tests::stdexec
