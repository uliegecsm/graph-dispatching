#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/graph/adoption/UserCode.hpp"

/**
 * @addtogroup unittests
 *
 * Adoption diamond without underlying @c Kokkos::Graph (eager)
 * ------------------------------------------------------------
 *
 * Create a simple diamond-like graph (see @ref intertwine/test_outlook.cpp) but do not map it
 * to any underlying @c Kokkos::Graph.
 *
 * Instead, make it like if the code was not really using P2300 by enforcing
 * eager execution. The code should look 99% like the one in @ref adoption/test_graph.cpp though.
 *
 * The test can be found in @ref adoption/test_regular.cpp.
 */

namespace tests::graph::adoption
{

/**
 * @test This test fakes a user calling library functions, but starting a chain
 *       that is not handled by a @c Kokkos::Graph.
 */
TEST(graph, adoption_regular)
{
    using execution_space = Kokkos::DefaultExecutionSpace;

    const execution_space exec {};

    decltype(auto) root = Kokkos::Experimental::just(exec);

    static_assert(std::same_as<decltype(root), const execution_space&>);

    decltype(auto) from_user = user_code(root);

    static_assert(std::same_as<decltype(from_user), const execution_space&>);

    Kokkos::Experimental::submit(exec, from_user);

    exec.fence();
}

} // namespace tests::graph::adoption
