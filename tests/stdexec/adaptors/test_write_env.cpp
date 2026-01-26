#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
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
 *  - @cite P3325R5
 *  - @cite P3284R4
 *  - https://github.com/NVIDIA/stdexec/blob/250f35737790392d666fd157e0af9be16d0c789f/test/exec/test_env.cpp
 *
 * The test can be found in @ref stdexec/adaptors/test_write_env.cpp.
 */

namespace tests::stdexec::adaptors {

//! Custom property (forwarding).
constexpr struct CustomProperty
    : public ::stdexec::__query<CustomProperty>
    , ::stdexec::forwarding_query_t {
    using ::stdexec::__query<CustomProperty>::operator();
} my_custom_property{};

//! Yet another custom property.
constexpr struct YetAnotherCustomProperty : public ::stdexec::__query<YetAnotherCustomProperty> {
} my_yet_another_custom_property{};

/// @test Check that we can build an environment with a custom property and query it at compile time.
TEST(Environment, build_env_constexpr) {
    constexpr auto env = ::stdexec::prop{my_custom_property, 123};

    using expt_env_t = ::stdexec::__env::prop<tests::stdexec::adaptors::CustomProperty, int>;

    static_assert(std::same_as<decltype(env), const expt_env_t>);

    constexpr auto value = my_custom_property(env);

    static_assert(std::same_as<decltype(value), const int>);

    static_assert(value == 123);
}

/// @test Check that we can build an environment with many properties.
TEST(Environment, build_env_constexpr_many_props) {
    constexpr auto env = ::stdexec::env{
        ::stdexec::prop{            my_custom_property, 123},
        ::stdexec::prop{my_yet_another_custom_property, 456}
    };

    constexpr auto value_cp = my_custom_property(env);
    constexpr auto value_yacp = my_yet_another_custom_property(env);

    static_assert(std::same_as<decltype(value_cp), const int>);
    static_assert(std::same_as<decltype(value_yacp), const int>);

    static_assert(value_cp == 123);
    static_assert(value_yacp == 456);
}

//! @test This is testing the issue https://github.com/NVIDIA/stdexec/issues/1606.
TEST(Environment, build_env_constexpr_single_prop) {
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ < 15)
    constexpr auto env = ::stdexec::prop{my_custom_property, 123};
#else
    constexpr auto env = ::stdexec::env{
        ::stdexec::prop{my_custom_property, 123}
    };
#endif

    static_assert(my_custom_property(env) == 123);
}

struct SomeRuntimeState {
    std::string label;
};

//! @test Check that we can build an environment with a custom runtime property and query it.
TEST(Environment, build_env_runtime) {
    const auto env = ::stdexec::prop{my_custom_property, SomeRuntimeState{"hello darkness my old friend"}};

    using expt_env_t = ::stdexec::__env::prop<tests::stdexec::adaptors::CustomProperty, SomeRuntimeState>;

    static_assert(std::same_as<decltype(env), const expt_env_t>);

    const auto& value = my_custom_property(env);

    static_assert(std::same_as<decltype(value), const SomeRuntimeState&>);

    ASSERT_EQ(value.label, "hello darkness my old friend");
}

/**
 * @test Check that we can use @c stdexec::let_value to read from the environment (for debugging what's in there).
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/250f35737790392d666fd157e0af9be16d0c789f/test/nvexec/let_value.cpp#L132-L149.
 */
TEST(Environment, let_value_read_env) {
    std::ostringstream out;

    auto env = ::stdexec::prop{my_custom_property, SomeRuntimeState{"try catch me if you can"}};

    ::stdexec::sender auto chain = ::stdexec::just()
                                 | ::stdexec::let_value([] { return ::stdexec::read_env(my_custom_property); })
                                 | ::stdexec::then([&out](const SomeRuntimeState& state) { out << state.label; });

    ::stdexec::sync_wait(std::move(chain) | ::stdexec::write_env(std::move(env))); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(out.str(), "try catch me if you can");
}

} // namespace tests::stdexec::adaptors
