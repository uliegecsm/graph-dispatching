#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

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
    view_t data(Kokkos::view_alloc(exec, "data"));

    //! Define the graph. Use a simple syntax.
    using policy_t = Kokkos::RangePolicy<execution_space>;

    constexpr int value_A = 5, value_B = 42, value_C = 156, value_D = 453;

    auto graph = Kokkos::Experimental::create_graph<execution_space>(exec);

    auto root = Kokkos::Impl::GraphAccess::create_root_ref(graph);

    auto node_A = root.then_parallel_for(
        policy_t(0, size),
        AddValueOffset{.data = data, .value = value_A});

    auto node_B = node_A.then_parallel_for(
        policy_t(0, size / 2),
        AddValueOffset{.data = data, .value = value_B});
    auto node_C = node_A.then_parallel_for(
        policy_t(size / 2, size),
        AddValueOffset{.data = data, .value = value_C});

    auto node_D = Kokkos::Experimental::when_all(node_B, node_C).then_parallel_for(
        policy_t(0, size),
        AddValueOffset{.data = data, .value = value_D});

    //! Execute the graph and check results.
    graph.submit(exec);

    ASSERT_IT_WENT_FINE
}

} // namespace tests::graph::diamond
