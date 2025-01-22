#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "tests/graph/then/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * A @c then node with @c Kokkos
 * -----------------------------
 *
 * Create a graph with a @c then node using @c Kokkos.
 *
 * The test can be found in @ref then/test_kokkos.cpp.
 */

namespace tests::graph::then
{

//! @test A graph with a @c then node using @c Kokkos.
TEST(graph, then_kokkos)
{
    //! Use @c Kokkos::DefaultExecutionSpace because we can synchronize in this setup.
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space    = typename execution_space::memory_space;

    //! Definition of the data type.
    using view_t = Kokkos::View<int, memory_space>;

    //! Get some execution context.
    const auto execs = Kokkos::Experimental::partition_space(execution_space{}, 1, 1, 1);

    //! Initialize the data.
    view_t data(Kokkos::view_alloc(execs.at(0), "data"));

    //! Define the graph. Use a simple syntax.
    auto graph = Kokkos::Experimental::create_graph<execution_space>(execs.at(0), [&](const auto& root) {
        root.then("node A",              ThenFunctor<view_t>{.data = data})
            .then("node B",              ThenFunctor<view_t>{.data = data})
            .then("node C", execs.at(1), ThenFunctor<view_t>{.data = data});
    });

    //! Execute the graph and check results.
    graph.submit(execs.at(2));
    execs.at(2).fence();

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_EQ(mirror(), 3);
}

} // namespace tests::graph::then
