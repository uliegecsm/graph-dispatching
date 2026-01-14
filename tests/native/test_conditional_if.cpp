#include <filesystem>

#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Play with @c Cuda conditional @c if node
 * ----------------------------------------
 *
 * This test shows how a @c Cuda conditional @c if node could be used.
 *
 * The test can be found in @ref native/test_conditional_if.cpp.
 */

namespace tests::cuda
{

/**
 * @brief Functor that will set the conditional to either @c true or @c false.
 *
 * This is a regular functor, but it needs to set a @c Cuda handle to the conditional
 * value with @c cudaGraphSetConditional.
 */
struct WorkThatDrivesTheConditional
{
    bool decision;

    cudaGraphConditionalHandle handle;

    /// @note @c cudaGraphSetConditional is a pure device function.
    ///       Therefore, we cannot decorate our call operator with @ref KOKKOS_FUNCTION.
    __device__
    void operator()(const int) const {
        cudaGraphSetConditional(handle, decision);
    }
};

class cuda : public ::testing::TestWithParam<bool> {};

//! @test Check how a @c Cuda conditional @c if node can be used.
TEST_P(cuda, conditional_if)
{
    using view_t    = View<int>;
    using functor_t = MyFunctor<view_t>;

    constexpr size_t size = 1;

    const Stream stream;

    const view_t data(stream, size);

    const Graph graph;

    cudaGraphConditionalHandle handle;
    CHECK_CALL(cudaGraphConditionalHandleCreate(&handle, graph.graph));

    const WorkThatDrivesTheConditional functor_decision{
        .decision = this->GetParam(),
        .handle   = handle
    };
    GraphNodeKernel node_decision(functor_decision, 1);
    node_decision.add(graph);

    GraphNodeConditionalIf conditional(handle);
    conditional.add(graph, {node_decision}); // NOLINT(cppcoreguidelines-slicing)

    const functor_t functor_if{.data = data};
    GraphNodeKernel node_if(functor_if, 1);
    node_if.add(conditional.get());

    const GraphExecutable graph_exec(graph);

    graph_exec.submit(stream);

    const auto mirror = data.get_host_copy(stream);

    ASSERT_EQ(mirror.at(0), this->GetParam() ? 1 : 0);

    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_conditional_if.dot").c_str(), cudaGraphDebugDotFlagsVerbose);
}

INSTANTIATE_TEST_SUITE_P(
    Decide,
    cuda,
    ::testing::Values(true, false)
);

} // namespace tests::cuda
