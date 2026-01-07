#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Constrained environment
 * -----------------------
 *
 * This test will fail at compile time because it tries to add a property of the wrong type.
 * According to @cite P3325R5 (Section 4.2.4, "Should Environments Be Constrained?"),
 * it must fail.
 *
 * The test can be found in @ref stdexec/adaptors/test_write_env.fail.cpp.
 */

namespace tests::stdexec::adaptors
{

template <typename Env, typename Query>
concept has_query = requires (const Env& env) { env.query(Query{}); };

//! Constrained property. Only @c double is accepted.
struct get_double_property_t
{
    template <typename Env> requires has_query<Env, get_double_property_t>
    decltype(auto) operator()(const Env& env) const
    {
        static_assert(std::same_as<decltype(env.query(*this)), double>,
            "The 'get_double_property_t' query must return a double.");

        return env.query(*this);
    }
};

//! The tag to use in the code for @ref get_double_property_t.
constexpr get_double_property_t get_double_property {};

//! @test A constrained property enforces the type.
TEST(Environment, constrained_property_enforces_type)
{
    constexpr auto env = ::stdexec::env{::stdexec::prop{get_double_property, char(123)}};
}

} // namespace tests::stdexec::adaptors
