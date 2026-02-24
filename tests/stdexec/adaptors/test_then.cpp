#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/utils/Counter.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::then
 * --------------------------
 *
 * This group of tests check the behavior of @c stdexec::then.
 *
 * The tests can be found in @ref tests/stdexec/adaptors/test_then.cpp.
 */

namespace tests::stdexec::adaptors {

//! @test Check how many times @ref tests::utils::Counter is constructed/moved/copied.
TEST(ThenTest, copies_and_moves) {
    ::tests::utils::Counter::reset();

    auto sndr = ::stdexec::just() | ::stdexec::then(::tests::utils::Counter{});

    ASSERT_EQ(::tests::utils::Counter::default_constructions.load(), 1);
    ASSERT_EQ(::tests::utils::Counter::copy_assignments.load(), 0);
    ASSERT_EQ(::tests::utils::Counter::copy_constructions.load(), 0);
    ASSERT_EQ(::tests::utils::Counter::move_assignments.load(), 0);
    ASSERT_EQ(::tests::utils::Counter::move_constructions.load(), 4);

    //! It will copy once.
    ::stdexec::sync_wait(sndr);

    ASSERT_EQ(::tests::utils::Counter::default_constructions.load(), 1);
    ASSERT_EQ(::tests::utils::Counter::copy_assignments.load(), 0);
    ASSERT_EQ(::tests::utils::Counter::copy_constructions.load(), 1);
    ASSERT_EQ(::tests::utils::Counter::move_assignments.load(), 0);
    ASSERT_EQ(::tests::utils::Counter::move_constructions.load(), 4);

    //! It will move (more than once).
    ::stdexec::sync_wait(std::move(sndr));

    ASSERT_EQ(::tests::utils::Counter::default_constructions.load(), 1);
    ASSERT_EQ(::tests::utils::Counter::copy_assignments.load(), 0);
    ASSERT_EQ(::tests::utils::Counter::copy_constructions.load(), 1);
    ASSERT_EQ(::tests::utils::Counter::move_assignments.load(), 0);
    ASSERT_EQ(::tests::utils::Counter::move_constructions.load(), 7);
}

} // namespace tests::stdexec::adaptors
