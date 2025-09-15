#include <filesystem>

#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Compose graphs together using the subgraph feature
 * --------------------------------------------------
 *
 * This test shows how graphs can be combined to create new graphs, using the
 * notion of "child graph node" (see @c cudaGraphAddChildGraphNode).
 *
 * The test can be found in @ref native/test_subgraph.cpp.
 */

namespace tests::cuda
{

#define CREATE_KERNEL_NODE(__name__, __data__, __graph__, ...)          \
    const functor_t functor_##__name__ { .data = __data__ };            \
    GraphNodeKernel node_##__name__(functor_##__name__, __data__.size); \
    node_##__name__.add(__graph__ __VA_OPT__(,) __VA_ARGS__);

//! @test Compose a graph using 2 subgraphs.
TEST(cuda, subgraph)
{
    using value_t   = int;
    using view_t    = View<value_t>;
    using functor_t = MyFunctor<view_t, false>;

    //! Initialize data.
    const Stream stream;

    const view_t data(stream, 3);

    const view_t data_A      (2, data.buffer    );
    const view_t data_A_elt_0(1, data.buffer    );
    const view_t data_A_elt_1(1, data.buffer + 1);
    const view_t data_B      (1, data.buffer + 2);

    //! Create subgraph A.
    const Graph subgraph_A;

    CREATE_KERNEL_NODE(a, data_A,       subgraph_A)
    CREATE_KERNEL_NODE(b, data_A_elt_0, subgraph_A, {node_a})
    CREATE_KERNEL_NODE(c, data_A_elt_1, subgraph_A, {node_a})
    CREATE_KERNEL_NODE(d, data_A,       subgraph_A, {node_b, node_c})

    //! Create subgraph B.
    const Graph subgraph_B;

    CREATE_KERNEL_NODE(e, data_B, subgraph_B)
    CREATE_KERNEL_NODE(f, data_B, subgraph_B, {node_e})

    //! Compose subgraphs together.
    const Graph graph;

    CREATE_KERNEL_NODE(g, data, graph)

    const GraphNode subgraph_A_embedded = subgraph_A.add(graph, {node_g});
    const GraphNode subgraph_B_embedded = subgraph_B.add(graph, {node_g});

    CREATE_KERNEL_NODE(h, data, graph, {subgraph_A_embedded, subgraph_B_embedded})

    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_subgraph.dot").c_str(), PREFIXED_API(GraphDebugDotFlagsVerbose));

    //! Execute the graph.
    const GraphExecutable graph_exec(graph);

    graph_exec.submit(stream);

    //! Check results.
    ASSERT_EQ(data.get_host_copy(stream), (std::vector<value_t>{5, 5, 4}));
}

} // namespace tests::cuda
