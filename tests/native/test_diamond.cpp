#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Diamond graph with @c Cuda API wrappers
 * ---------------------------------------
 *
 * This test builds a simple diamond graph with @c Cuda API wrappers.
 *
 * The test can be found in @ref native/test_diamond.cpp.
 */

namespace tests::cuda
{

//! @test Build a diamond graph with @c Cuda API wrappers.
TEST(cuda, diamond_graph)
{
    using view_t    = View<int>;
    using functor_t = MyFunctor<view_t>;

    constexpr size_t size = 1;

    const Stream stream;

    const view_t data(stream, size);

    const Graph graph;

    const functor_t functor_A {.data = data};
    GraphNodeKernel node_A(functor_A, size);
    node_A.add(graph);

    const functor_t functor_B{.data = data};
    GraphNodeKernel node_B(functor_B, 1);
    node_B.add(graph, {node_A}); // NOLINT(cppcoreguidelines-slicing)

    const functor_t functor_C{.data = data};
    GraphNodeKernel node_C(functor_C, 1);
    node_C.add(graph, {node_A}); // NOLINT(cppcoreguidelines-slicing)

    const functor_t functor_D{.data = data};
    GraphNodeKernel node_D(functor_D, 1);
    node_D.add(graph, {node_B, node_C}); // NOLINT(cppcoreguidelines-slicing)

    const GraphExecutable graph_exec(graph);

    graph_exec.submit(stream);

    const auto mirror = data.get_host_copy(stream);

    ASSERT_THAT(mirror, ::testing::ElementsAre(4));

    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_diamond.dot").c_str(), PREFIXED_API(GraphDebugDotFlagsVerbose));
}

} // namespace tests::cuda
