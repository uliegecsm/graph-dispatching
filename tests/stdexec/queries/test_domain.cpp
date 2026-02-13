#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

/**
 * @addtogroup unittests
 *
 * Tests related to the domain query
 * ---------------------------------
 *
 * This group of tests check behavior related to the domain query.
 *
 * The tests can be found in @ref tests/stdexec/queries/test_domain.cpp.
 */

namespace tests::stdexec::queries {

//! @test Check @c ::stdexec::__common_domain_t.
TEST(Domain, common_domain_traits) {
    struct MyDomain { };
    struct MyOtherDomain { };
    struct MyDerivedDomain : public MyDomain { };

    static_assert(std::same_as<::stdexec::__common_domain_t<MyDomain, MyDomain>, MyDomain>);
    static_assert(::stdexec::__has_common_domain<MyDomain, MyDomain>);

    static_assert(std::same_as<
                  ::stdexec::__common_domain_t<MyDomain, MyOtherDomain>,
                  ::stdexec::indeterminate_domain<MyOtherDomain, MyDomain>
    >);
    static_assert(::stdexec::__has_common_domain<MyDomain, MyOtherDomain>);

    /**
     * The common domain of a base domain and a derived domain is the base domain. The reason is that the implementation
     * of @c ::stdexec::__common_domain_t relies on @c std::common_type_t.
     *
     * See https://github.com/NVIDIA/stdexec/blob/8cfc3f1983d3521b341864074123281011f998c1/include/stdexec/__detail/__domain.hpp#L157-L161.
     */
    static_assert(std::same_as<::stdexec::__common_domain_t<MyDomain, MyDerivedDomain>, MyDomain>);
    static_assert(std::same_as<std::common_type_t<MyDomain, MyDerivedDomain>, MyDomain>);
}

} // namespace tests::stdexec::queries
