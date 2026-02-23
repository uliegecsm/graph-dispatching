#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

/**
 * @addtogroup unittests
 *
 * Execution policy traits
 * -----------------------
 *
 * This group of tests check the traits of @c Kokkos execution policies.
 *
 * The tests can be found in @ref tests/kokkos/test_ExecutionPolicy.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos
{

//! @test Check nothrow traits of @c Kokkos::RangePolicy.
constexpr bool test_range_policy_traits() {
    using policy_t = Kokkos::RangePolicy<execution_space>;

    static_assert(!std::is_nothrow_constructible_v<policy_t, int, int>);

    static_assert(std::is_nothrow_move_constructible_v<policy_t>);
    static_assert(std::is_nothrow_move_assignable_v<policy_t>);

    static_assert(std::is_nothrow_copy_constructible_v<policy_t>);
    static_assert(std::is_nothrow_copy_assignable_v<policy_t>);

    return true;
}
static_assert(test_range_policy_traits());

//! @test Check behavior of default @c Kokkos::RangePolicy.
TEST(RangePolicy, default) {
    auto policy = Kokkos::RangePolicy(42, 666);

    static_assert(std::same_as<decltype(policy), Kokkos::RangePolicy<>>);

    ASSERT_EQ(policy.space(), execution_space{});
}

} // namespace tests::kokkos
