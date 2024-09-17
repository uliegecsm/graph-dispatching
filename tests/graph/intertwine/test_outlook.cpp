#include "gtest/gtest.h"

#include "tests/graph/Kokkos_Graph_Execution.hpp"
#include "tests/graph/diamond/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Intertwining with P2300-flavored @c Kokkos
 * ------------------------------------------
 *
 * Create an intertwining graph with @c Kokkos *à la* P2300, inspired by the diamond case (see @ref diamond/test_outlook.cpp).
 *
 * The test can be found in @ref intertwine/test_outlook.cpp.
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
decltype(auto) library(Sender&& input, ViewType data)
{
    //! @todo We need to expose our @c operator|, otherwise the compiler can't find a match.
    using Kokkos::Experimental::graph::details::operator|;

    auto continued = std::forward<Sender>(input) | Kokkos::Experimental::graph::split();

    using policy_t = Kokkos::RangePolicy<typename ViewType::execution_space>;

    auto one = input | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, data.size() / 2),
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_B});

    auto two = input | Kokkos::Experimental::graph::parallel_for(
        policy_t(data.size() / 2, data.size()),
        diamond::AddValueOffset{.data = std::move(data), .value = diamond::Values::value_C});

    return Kokkos::Experimental::when_all(std::move(one), std::move(two));
}

//! @test Intertwine graph using @c Kokkos.
TEST(graph, intertwine_outlook)
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

    auto root = Kokkos::Experimental::graph::just(exec);

    auto node_A = root | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, size),
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_A});

    auto from_library = library(node_A, data);

    auto node_D = from_library | Kokkos::Experimental::graph::parallel_for(
        policy_t(0, size),
        diamond::AddValueOffset{.data = data, .value = diamond::Values::value_D});

    //! Execute the graph and check results.
    Kokkos::Experimental::graph::submit(exec, node_D);

    ASSERT_TRUE(diamond::check_data(exec, data));
}

} // namespace tests::graph::intertwine
