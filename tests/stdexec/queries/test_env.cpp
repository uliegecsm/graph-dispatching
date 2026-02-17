#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests related to environment queries
 * ------------------------------------
 *
 * This group of tests check behavior related to environment queries.
 *
 * The tests can be found in @ref tests/stdexec/queries/test_env.cpp.
 */

namespace tests::stdexec::queries {

constexpr struct FwdFoo
    : ::stdexec::__query<FwdFoo>
    , ::stdexec::forwarding_query_t {
    using ::stdexec::__query<FwdFoo>::operator();
} fwd_foo{};

constexpr struct FwdBar
    : ::stdexec::__query<FwdBar>
    , ::stdexec::forwarding_query_t {
    using ::stdexec::__query<FwdBar>::operator();
} fwd_bar{};

constexpr struct Foo : ::stdexec::__query<Foo> {
} foo{};

constexpr struct Bar : ::stdexec::__query<Bar> {
} bar{};

//! @test @c stdexec::prop within an @c stdexec::env makes it queryable for the property, no matter if it's a forwarding query or not.
constexpr bool test_prop_makes_env_queryable() {
    auto env = ::stdexec::env{
        ::stdexec::prop{fwd_foo, 42.},
        ::stdexec::prop{    foo, 'F'}
    };

    return fwd_foo(env) == 42. && foo(env) == 'F';
}
static_assert(test_prop_makes_env_queryable());

//! @test @c stdexec::prop within nested @c stdexec::env makes it queryable for the property, no matter if it's a forwarding query or not.
constexpr bool test_prop_makes_nested_env_queryable() {
    auto env = ::stdexec::env{
        ::stdexec::env{::stdexec::prop{fwd_foo, 42.}, ::stdexec::prop{foo, 'F'}}
    };

    return fwd_foo(env) == 42. && foo(env) == 'F';
}
static_assert(test_prop_makes_nested_env_queryable());


/**
 * @test A @c stdexec::prop within nested @c stdexec::env with an additional @c stdexec::prop alongside the nested @c stdexec::env
 *       can be queried for.
 *
 * See also https://github.com/NVIDIA/stdexec/issues/1840.
 */
constexpr bool test_prop_makes_nested_env_queryable_only_fwd_query() {
    auto env = ::stdexec::env{
        ::stdexec::env{::stdexec::prop{fwd_foo, 42.}, ::stdexec::prop{foo, 'F'}},
        ::stdexec::prop{                          bar,                     31415}
    };

    static_assert(::stdexec::__queryable_with<decltype(env), Foo>);
    return fwd_foo(env) == 42. && foo(env) == 'F' && bar(env) == 31415;
}
static_assert(test_prop_makes_nested_env_queryable_only_fwd_query());

//! @test Forwarding queries are always queryable, no matter the nesting of @c stdexec::env.
constexpr bool test_prop_deeply_nested_fwd_queries() {
    auto env = ::stdexec::env{::stdexec::env{::stdexec::env{
        ::stdexec::prop{fwd_bar, 42.}, ::stdexec::__env::__fwd{::stdexec::env{::stdexec::prop{fwd_foo, 'F'}}}}}};

    static_assert(!::stdexec::__queryable_with<decltype(env), Foo>);
    static_assert(!::stdexec::__queryable_with<decltype(env), Bar>);

    return fwd_foo(env) == 'F' && fwd_bar(env) == 42.;
}
static_assert(test_prop_deeply_nested_fwd_queries());

//! @test Runtime constructed objects are allowed (of course).
TEST(prop, runtime) {
    auto env = ::stdexec::env{
        ::stdexec::prop{foo, std::string("My runtime variable!")}
    };
    ASSERT_EQ(foo(env), "My runtime variable!");
}

} // namespace tests::stdexec::queries
