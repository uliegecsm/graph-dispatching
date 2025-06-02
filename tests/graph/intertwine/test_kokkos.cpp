#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "tests/graph/diamond/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Intertwining with @c Kokkos
 * ---------------------------
 *
 * Create an intertwining graph with @c Kokkos, inspired by the diamond case (see @ref diamond/test_kokkos.cpp).
 *
 * The test can be found in @ref intertwine/test_kokkos.cpp.
 */

namespace tests::graph::intertwine
{

/**
 * @brief Faking an external library call that adds its workloads to the chain.
 *
 * It adds 2 asynchronous workloads. The input sender is split, and both workloads are
 * joined and returned.
 */
template <typename Sender, typename ViewType>
decltype(auto) library(const Sender& input, ViewType data)
{
    using policy_t = Kokkos::RangePolicy<typename ViewType::execution_space>;

    auto one = input.then_parallel_for(
        policy_t(0, data.size() / 2),
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_B});

    policy_t policy(data.size() / 2, data.size());
    auto two = input.then_parallel_for(
        std::move(policy),
        diamond::AddValueOffset{.data = std::move(data), .value = diamond::Values::value_C});

    return Kokkos::Experimental::when_all(std::move(one), std::move(two));
}

//! @test Intertwine graph using @c Kokkos.
TEST(graph, intertwine_kokkos)
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
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_A});

    auto from_library = library(node_A, data);

    auto node_D = from_library.then_parallel_for(
        policy_t(0, size),
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_D});

    //! Execute the graph and check results.
    graph.submit(exec);

    ASSERT_TRUE(diamond::check_data(exec, data));
}

} // namespace tests::graph::intertwine
