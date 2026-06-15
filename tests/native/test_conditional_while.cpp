#include <filesystem>

#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Play with @c Cuda conditional @c while node
 * -------------------------------------------
 *
 * This test shows how a @c Cuda conditional @c while node could be used.
 *
 * The test can be found in @ref native/test_conditional_while.cpp.
 */

namespace tests::cuda
{

/**
 * @brief Functor that will set the conditional to either @c true or @c false.
 *
 * This is a regular functor, but it needs to set a @c Cuda handle to the conditional
 * value with @c cudaGraphSetConditional.
 */
template <typename ViewType>
struct WorkThatDrivesTheConditionalWhile
{
    ViewType decision;

    cudaGraphConditionalHandle handle;

    /// @note @c cudaGraphSetConditional is a pure device function.
    ///       Therefore, we cannot decorate our call operator with @ref KOKKOS_FUNCTION.
    __device__
    void operator()(const int) const {
        if(decision(0) >= 42) cudaGraphSetConditional(handle, false);
    }
};

/**
 * @test Check how a @c Cuda conditional @c while node can be used.
 *
 * The first element of a device view is set to 0. In a while loop until it is equal to 42,
 * it will be incremented.
 *
 * In the @c while node, we have 2 nodes:
 *  1. a kernel that increments the data
 *  2. a kernel that sets the handle.
 */
TEST(cuda, conditional_while)
{
    using view_t    = View<int>;
    using functor_t = MyFunctor<view_t, false>;

    constexpr size_t size = 1;

    const Stream stream;

    const view_t data(stream, size);

    const Graph graph;

    cudaGraphConditionalHandle handle;
    CHECK_CALL(cudaGraphConditionalHandleCreate(&handle, graph.graph, 1, cudaGraphCondAssignDefault));

    GraphNodeConditionalWhile conditional(handle);
    conditional.add(graph, {});

    const functor_t work_f{.data = data};
    GraphNodeKernel work_n(work_f, 1);
    work_n.add(conditional.get(), {});

    const WorkThatDrivesTheConditionalWhile decision_f{
        .decision = data,
        .handle   = handle
    };
    GraphNodeKernel decision_n(decision_f, 1);
    decision_n.add(conditional.get(), {work_n}); // NOLINT(cppcoreguidelines-slicing)

    const GraphExecutable graph_exec(graph);

    graph_exec.submit(stream);

    const auto mirror = data.get_host_copy(stream);
    stream.fence();

    EXPECT_EQ(mirror.at(0), 42);

    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_conditional_while.dot").c_str(), cudaGraphDebugDotFlagsVerbose);
}

} // namespace tests::cuda
