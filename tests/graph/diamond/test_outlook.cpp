#include "gtest/gtest.h"

#include "tests/graph/Kokkos_Graph_Execution.hpp"
#include "tests/graph/diamond/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Diamond with P2300-flavored @c Kokkos
 * -------------------------------------
 *
 * Create a diamond graph with @c Kokkos *à la* P2300.
 *
 * The test can be found in @ref diamond/test_outlook.cpp.
 */

namespace tests::graph::diamond
{

//! @test Diamond graph using P2300-flavored @c Kokkos.
TEST(graph, diamond_outlook)
{
    //! Use @c Kokkos::DefaultExecutionSpace because we can synchronize in this setup.
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

    //! Definition of the data type.
    constexpr size_t size = 10;
    static_assert(size % 2 == 0);

    using view_t = Kokkos::View<int[size], memory_space>;

    //! Get some execution context.
    const execution_space exec {};

    //! Initialize the data.
    view_t data(Kokkos::view_alloc(exec, "data"));

    //! Define the graph. Use a simple syntax.
    using policy_t = Kokkos::RangePolicy<execution_space>;

    auto root = Kokkos::Experimental::graph::just(exec) | Kokkos::Experimental::graph::split();

    auto node_A = root | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, size),
        AddValueOffset{.data = data, .value = Values::value_A});

    auto node_B = node_A | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, size / 2),
        AddValueOffset{.data = data, .value = Values::value_B});
    auto node_C = node_A | Kokkos::Experimental::graph::parallel_for(
        policy_t(size / 2, size),
        AddValueOffset{.data = data, .value = Values::value_C});

    auto node_D = Kokkos::Experimental::when_all(node_B, node_C) | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, size),
        AddValueOffset{.data = data, .value = Values::value_D});

    //! Execute the graph and check results.
    Kokkos::Experimental::graph::submit(exec, node_D);

    ASSERT_TRUE(check_data(exec, data));
}

} // namespace tests::graph::diamond
