#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/ArborX/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Use @c ArborX::BoundingVolumeHierarchy withing a graph
 * ------------------------------------------------------
 *
 * Create a simple diamond-like graph (see @ref intertwine/test_outlook.cpp) and map it
 * to an underlying @c Kokkos::Graph.
 *
 * The test can be found in @ref ArborX/BoundingVolumeHierarchy/test_graph.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space    = typename execution_space::memory_space;

using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace tests::ArborX::BoundingVolumeHierarchy
{

//! @test Build the @c ArborX::BoundingVolumeHierarchy tree withing a graph.
TEST(BoundingVolumeHierarchy, constructor_in_graph)
{
    const execution_space      exec   {};
    const host_execution_space exec_h {};

    const auto boxes = create_boxes<memory_space>(exec, exec_h);

    auto root = Kokkos::Experimental::graph::create_graph(exec);

    /**
     * Problems encountered in moving the constructor to the graph:
     *  1. Allocations using @c Kokkos::view_alloc(exec,...) don't compile.
     *  2. @c Kokkos concepts like @c accessible aren't working if they don't receive execution space instances
     *     (they got a graph chain handler in this case).
     */
    // const ::ArborX::BoundingVolumeHierarchy tree(root, ::ArborX::Experimental::attach_indices(boxes));
}

//! @test Query the @c ArborX::BoundingVolumeHierarchy tree withing a graph.
TEST(BoundingVolumeHierarchy, query_in_graph)
{

}

} // namespace tests::ArborX::BoundingVolumeHierarchy
