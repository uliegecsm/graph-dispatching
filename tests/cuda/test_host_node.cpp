#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/cuda/APIWrappers_def.hpp"
#include "tests/cuda/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Play with @c Cuda graph host node
 * ---------------------------------
 *
 * This test shows how a @c Cuda graph host node can be added.
 *
 * The test can be found in @ref cuda/test_host_node.cpp.
 */

namespace tests::cuda
{

/**
 * @brief Dumb "host" functor. Note how the call operator does not expect any argument.
 *
 * @note The same restrictions for a host node apply as for @c cudaStreamAddCallback.
 *       For instance, no @c Cuda function call is permitted.
 */
struct MyHostFunctor
{
    double data;

    void operator()() const {
        std::cout << "Hello from " << __PRETTY_FUNCTION__ << ": data is " << data << std::endl;
    }
};

//! @test Check how a @c Cuda graph host node can be added.
TEST(cuda, host_node)
{
    using view_t      = View<int>;
    using functor_d_t = MyFunctor<view_t>;
    using functor_h_t = MyHostFunctor;

    constexpr size_t size = 1;

    Stream stream;

    view_t data(stream, size);

    Graph graph;

    functor_d_t functor_d{.data = data};
    GraphNodeKernel node_d(functor_d, 1);
    node_d.add(graph);

    GraphNodeHost<functor_h_t> node_h{functor_h_t {.data = 42.}};
    node_h.add(graph, {node_d});

    GraphExecutable graph_exec(graph);

    ::testing::internal::CaptureStdout();

    graph_exec.submit(stream);
    stream.fence();

    ASSERT_THAT(::testing::internal::GetCapturedStdout(), ::testing::HasSubstr(
        "Hello from void tests::cuda::MyHostFunctor::operator()() const: data is 42"
    ));

    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_host_node.dot").c_str(), cudaGraphDebugDotFlagsVerbose);
}

} // namespace tests::cuda
