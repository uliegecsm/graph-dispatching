#include <filesystem>

#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Play with @c Cuda graph node enable/disable feature
 * ---------------------------------------------------
 *
 * This test shows how a @c Cuda graph node can be enabled/disabled from the host
 * in-between graph submissions.
 *
 * The test can be found in @ref native/test_node_enable.cpp.
 */

namespace tests::cuda
{

class cuda : public ::testing::TestWithParam<std::vector<bool>> {};

//! @test Check how a @c Cuda graph node can be enabled/disabled.
TEST_P(cuda, node_enable)
{
    using view_t    = View<int>;
    using functor_t = MyFunctor<view_t>;

    constexpr size_t size = 1;

    const Stream stream;

    const view_t data(stream, size);

    const Graph graph;

    const functor_t functor{.data = data};
    GraphNodeKernel node(functor, 1);
    node.add(graph);

    const GraphExecutable graph_exec(graph);

    size_t expected = 0;

    for(const auto is_enabled : this->GetParam())
    {
        graph_exec.set_enabled(node, is_enabled);

        graph_exec.submit(stream);

        expected += is_enabled;
    }

    const auto mirror = data.get_host_copy(stream);
    stream.fence();

    ASSERT_EQ(mirror.at(0), expected);

    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_node_enable.dot").c_str(), PREFIXED_API(GraphDebugDotFlagsVerbose));
}

INSTANTIATE_TEST_SUITE_P(
    Enable,
    cuda,
    ::testing::Values(
        std::vector<bool>{true, true, false, false, true},
        std::vector<bool>{false, false},
        std::vector<bool>{true, false, true, false, true, false}
    )
);

} // namespace tests::cuda
