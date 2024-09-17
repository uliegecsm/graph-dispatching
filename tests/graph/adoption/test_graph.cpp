#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/graph/Kokkos_Graph_Execution.hpp"
#include "tests/graph/adoption/UserCode.hpp"

/**
 * @addtogroup unittests
 *
 * Adoption diamond with underlying @c Kokkos::Graph
 * -------------------------------------------------
 *
 * Create a simple diamond-like graph (see @ref intertwine/test_outlook.cpp) and map it
 * to an underlying @c Kokkos::Graph.
 *
 * The test can be found in @ref adoption/test_graph.cpp.
 */

namespace tests::graph::adoption
{

/**
 * @test This test fakes a user calling library functions, but starting a chain
 *       that is under-the-hood handled by a @c Kokkos::Graph.
 */
TEST(graph, adoption_graph)
{
    using execution_space = Kokkos::DefaultExecutionSpace;

    const execution_space exec {};

    decltype(auto) root = Kokkos::Experimental::graph::just(exec);

    decltype(auto) from_user = user_code(root);

    static_assert(Kokkos::Impl::is_specialization_of<decltype(from_user), Kokkos::Experimental::GraphNodeRef>::value);

    Kokkos::Experimental::graph::submit(exec, std::move(from_user));

    //! @todo This should not be a global fence.
    Kokkos::fence();
}

} // namespace tests::graph::adoption
