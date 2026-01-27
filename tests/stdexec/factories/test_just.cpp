#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wmissing-field-initializers")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::just
 * --------------------------
 *
 * This group of tests check the behavior of @c stdexec::just.
 *
 * The tests can be found in @ref tests/stdexec/factories/test_just.cpp.
 */

namespace tests::stdexec::factories {

/**
 * @test Use @c stdexec::just in a @c constexpr expression.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/5b0f1b1cfb7daa4f3798ee20aa643cd4f6e9e437/test/stdexec/algos/factories/test_just.cpp#L30.
 */
constexpr auto test_constexpr() noexcept {
    double placeholder = 0.;
    auto opstate =
        ::stdexec::connect(::stdexec::just(6.66), ::tests::stdexec::ValueReceiver{std::addressof(placeholder)});
    ::stdexec::start(opstate);
    return placeholder;
}
static_assert(test_constexpr() == 6.66);

//! @test Set the value channel with @c stdexec::just and retrieve the value after @c stdexec::sync_wait.
TEST(just, sync_wait_value) {
    const auto [value] = ::stdexec::sync_wait(::stdexec::just(double(42.))).value();

    static_assert(std::same_as<decltype(value), const double>);

    ASSERT_EQ(value, 42.);
}

} // namespace tests::stdexec::factories
