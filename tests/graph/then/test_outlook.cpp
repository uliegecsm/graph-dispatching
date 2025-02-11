#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/Functors.hpp"

/**
 * @addtogroup unittests
 *
 * A @c then node with P2300-flavored @c Kokkos
 * --------------------------------------------
 *
 * Create a graph with a @c then node using @c Kokkos *à la* P2300.
 *
 * The test can be found in @ref then/test_outlook.cpp.
 */

namespace tests::graph::then
{

//! @test A graph with a @c then node using P2300-flavored @c Kokkos.
TEST(graph, then_outlook)
{
    //! @todo We need to expose our @c operator|, otherwise the compiler can't find a match.
    using Kokkos::Experimental::graph::details::operator|;

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
    auto root = Kokkos::Experimental::graph::create_graph(execs.at(0));
    auto chain = root
                | Kokkos::Experimental::graph::then("node A",              ThenFunctor<view_t>{.data = data})
                | Kokkos::Experimental::graph::then("node B",              ThenFunctor<view_t>{.data = data})
                | Kokkos::Experimental::graph::then("node C", execs.at(1), ThenFunctor<view_t>{.data = data});

    //! Execute the graph and check results.
    Kokkos::Experimental::graph::submit(execs.at(2), std::move(chain));
    execs.at(2).fence();

    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

    ASSERT_EQ(mirror(), 3);
}

} // namespace tests::graph::then
