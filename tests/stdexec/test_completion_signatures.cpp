#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Completion signatures
 * ---------------------
 *
 * This group of tests check the behavior of @c stdexec tools to transform
 * completion signatures.
 *
 * The tests can be found in @ref tests/stdexec/test_completion_signatures.cpp.
 */

namespace tests::stdexec {

template <typename T, typename... Args>
constexpr void check_sigs() {
    static_assert(std::same_as<T, ::stdexec::completion_signatures<Args...>>);
}

//! @test Check that @c stdexec::transform_completion_signatures preserves the input completion signatures when nothing else is provided.
TEST(completion_signatures, transform) {
    using cs_empty_t = ::stdexec::completion_signatures<>;
    using cs_value_t = ::stdexec::completion_signatures<::stdexec::set_value_t(double)>;
    using cs_error_t = ::stdexec::completion_signatures<::stdexec::set_error_t(std::exception_ptr)>;
    using cs_stopped_t = ::stdexec::completion_signatures<::stdexec::set_stopped_t()>;

    check_sigs<::stdexec::transform_completion_signatures<cs_empty_t>>();
    check_sigs<::stdexec::transform_completion_signatures<cs_value_t>, ::stdexec::set_value_t(double)>();
    check_sigs<::stdexec::transform_completion_signatures<cs_error_t>, ::stdexec::set_error_t(std::exception_ptr)>();
    check_sigs<::stdexec::transform_completion_signatures<cs_stopped_t>, ::stdexec::set_stopped_t()>();
}

//! @test Check the behavior of @c stdexec::transform_completion_signatures_of.
TEST(completion_signatures, transform_of) {
    using sndr_int_t = decltype(::stdexec::just(int(42)));
    using sndr_double_t = decltype(::stdexec::just(double(3.14)));
    using sndr_eptr_t = decltype(::stdexec::just_error(std::exception_ptr{}));
    using sndr_stopped_t = decltype(::stdexec::just_stopped());

    using cs_int_t = ::stdexec::transform_completion_signatures_of<sndr_int_t>;
    using cs_double_t = ::stdexec::transform_completion_signatures_of<sndr_double_t>;
    using cs_eptr_t = ::stdexec::transform_completion_signatures_of<sndr_eptr_t>;
    using cs_stopped_t = ::stdexec::transform_completion_signatures_of<sndr_stopped_t>;

    check_sigs<cs_int_t, ::stdexec::set_value_t(int)>();
    check_sigs<cs_double_t, ::stdexec::set_value_t(double)>();
    check_sigs<cs_eptr_t, ::stdexec::set_error_t(std::exception_ptr)>();
    check_sigs<cs_stopped_t, ::stdexec::set_stopped_t()>();
}

} // namespace tests::stdexec
