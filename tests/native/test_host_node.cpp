#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Play with @c Cuda graph host node
 * ---------------------------------
 *
 * This test shows how a @c Cuda graph host node can be added.
 *
 * The test can be found in @ref native/test_host_node.cpp.
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

    const Stream stream;

    const Graph graph;

    const functor_d_t functor_d{.data = view_t{stream, size}};
    GraphNodeKernel node_d(functor_d, 1);
    node_d.add(graph);

    GraphNodeHost<functor_h_t> node_h{functor_h_t {.data = 42.}};
    node_h.add(graph, {node_d}); // NOLINT(cppcoreguidelines-slicing)

    const GraphExecutable graph_exec(graph);

    ::testing::internal::CaptureStdout();

    graph_exec.submit(stream);
    stream.fence();

    ASSERT_THAT(::testing::internal::GetCapturedStdout(), ::testing::HasSubstr(
        "Hello from void tests::cuda::MyHostFunctor::operator()() const: data is 42"
    ));

    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_host_node.dot").c_str(), PREFIXED_API(GraphDebugDotFlagsVerbose));
}

} // namespace tests::cuda
