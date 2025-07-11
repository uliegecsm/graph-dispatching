#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
#include "exec/env.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::write_env
 * -------------------------------
 *
 * This group of tests check the behavior of @c stdexec::write_env.
 *
 * See also:
 *  - (A Utility for Creating Execution Environments)[https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3325r2.html]
 *  - (@c finally, @c write_env, and @c unstoppable Sender Adaptors)[https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3284r1.html]
 *
 * The test can be found in @ref stdexec/adaptors/test_write_env.cpp.
 */

namespace tests::stdexec::adaptors
{

//! Custom property.
struct CustomProperty
{
    template <typename Env> requires ::stdexec::tag_invocable<CustomProperty, Env>
    constexpr decltype(auto) operator()(Env&& env) const {
        return ::stdexec::tag_invoke(*this, std::forward<Env>(env));
    }
};

constexpr CustomProperty my_custom_property {};

/// @test Check that we can build an environment with a custom property and query it at compile time.
/// @todo Make it a pure compile-time test when https://github.com/NVIDIA/stdexec/issues/1569 is addressed.
TEST(Environment, make_env_constexpr)
{
    const auto env = ::exec::make_env(::stdexec::prop{my_custom_property, 123});

    static_assert(std::same_as<decltype(env), const ::stdexec::__env::prop<tests::stdexec::adaptors::CustomProperty, int>>);

    const auto& value = my_custom_property(env);

    static_assert(std::same_as<decltype(value), const int&>);

    ASSERT_EQ(value, 123);
}

struct SomeRuntimeState
{
    std::string label;
};

//! @test Check that we can build an environment with a custom runtime property and query it.
TEST(Environment, make_env_runtime)
{
    const auto env = ::exec::make_env(::stdexec::prop{my_custom_property, SomeRuntimeState{"hello darkness my old friend"}});

    static_assert(std::same_as<decltype(env), const ::stdexec::__env::prop<tests::stdexec::adaptors::CustomProperty, SomeRuntimeState>>);

    const auto& value = my_custom_property(env);

    static_assert(std::same_as<decltype(value), const SomeRuntimeState&>);

    ASSERT_EQ(value.label, "hello darkness my old friend");
}

//! @test Check that we can use @c stdexec::let_value to read from the environment (for debugging what's in there).
TEST(Environment, let_value_read_env)
{
    std::ostringstream out;

    auto env = ::exec::make_env(
        ::stdexec::prop{my_custom_property, SomeRuntimeState{"try catch me if you can"}}
    );

    ::stdexec::sender auto chain = ::stdexec::just()
        | ::stdexec::let_value([] { return ::stdexec::read_env(my_custom_property); })
        | ::stdexec::then([&out](const SomeRuntimeState& state) {
            out << state.label;
        });
 
    ::stdexec::sync_wait(std::move(chain) | ::exec::write_env(std::move(env))); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(out.str(), "try catch me if you can");
}

} // namespace tests::stdexec::adaptors
