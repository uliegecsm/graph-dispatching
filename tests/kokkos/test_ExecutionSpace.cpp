#include <iostream>

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos {
/**
 * @test Check that execution space instances compare equal or different when expected.
 *
 * This is of uttermost importance for the defaulted graph implementation, see
 * https://github.com/kokkos/kokkos/blob/df2e3023dc1832a3ca4670ca8f81ae9fa8ac560d/core/src/impl/Kokkos_Default_GraphNode_Impl.hpp#L134-L139.
 */
TEST(ExecutionSpace, compare_different) {
    const execution_space defaulted{};

    const auto [instance_A, instance_B] = Kokkos::Experimental::partition_space(defaulted, 1, 1);

    std::cout << "Defaulted instance ID: " << defaulted.impl_instance_id() << std::endl;
    std::cout << "Instance A         ID: " << instance_A.impl_instance_id() << std::endl;
    std::cout << "Instance B         ID: " << instance_B.impl_instance_id() << std::endl;

    ASSERT_EQ(execution_space{}, defaulted);

    //! For @c Kokkos::OpenMP, see https://github.com/kokkos/kokkos/commit/a09c6ce45655f37bedf767d68ff42b7382ba89e7.
#if defined(KOKKOS_ENABLE_OPENMP)
    if constexpr (std::same_as<execution_space, Kokkos::OpenMP>) {
        ASSERT_EQ(defaulted, instance_A);
        ASSERT_EQ(defaulted, instance_B);
        ASSERT_EQ(instance_A, instance_B);
    } else {
#endif
        ASSERT_NE(defaulted, instance_A);
        ASSERT_NE(defaulted, instance_B);
        ASSERT_NE(instance_A, instance_B);
#if defined(KOKKOS_ENABLE_OPENMP)
    }
#endif
}

} // namespace tests::kokkos
