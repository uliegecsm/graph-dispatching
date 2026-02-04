#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "tests/Functors.hpp"
#include "tests/graph/diamond/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Diamond with @c Kokkos
 * ----------------------
 *
 * Create a diamond graph with @c Kokkos.
 *
 * The test can be found in @ref diamond/test_kokkos.cpp.
 */

namespace tests::graph::diamond
{

//! @test Diamond graph using @c Kokkos.
TEST(graph, diamond_kokkos)
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
    const view_t data(Kokkos::view_alloc(exec, "data"));

    //! Define the graph. Use a simple syntax.
    using policy_t = Kokkos::RangePolicy<execution_space>;

    const Kokkos::Experimental::Graph<execution_space> graph(exec);

    auto node_A = graph.root_node().then_parallel_for(
        policy_t(0, size),
        tests::AddValueOffset<view_t>{.data = data, .value = Values::value_A});

    auto node_B = node_A.then_parallel_for(
        policy_t(0, size / 2),
        tests::AddValueOffset<view_t>{.data = data, .value = Values::value_B});
    auto node_C = node_A.then_parallel_for(
        policy_t(size / 2, size),
        tests::AddValueOffset<view_t>{.data = data, .value = Values::value_C});

    auto node_D = Kokkos::Experimental::when_all(node_B, node_C).then_parallel_for(
        policy_t(0, size),
        tests::AddValueOffset<view_t>{.data = data, .value = Values::value_D});

    //! Execute the graph and check results.
    graph.submit(exec);

    ASSERT_TRUE(check_data(exec, data));
}

} // namespace tests::graph::diamond
