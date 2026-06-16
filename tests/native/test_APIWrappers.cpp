#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"

/**
 * @addtogroup unittests
 *
 * Simple test of the @c Cuda API wrappers
 * ---------------------------------------
 *
 * This group of tests check the behavior of the @c Cuda API wrappers provided in
 * @ref native/APIWrappers.hpp.
 *
 * The tests can be found in @ref native/test_APIWrappers.cpp.
 */

namespace tests::cuda
{

struct Dummy
{
    KOKKOS_FUNCTION
    void operator()(const int) const {}
};

//! @test Check that @ref tests::cuda::Stream works as expected.
TEST(APIWrappers, stream)
{
    const Stream stream;

    ASSERT_NE(stream.stream, nullptr);

#if defined(GRAPH_DISPATCHING_ENABLE_CUDA)
    unsigned long long stream_id;
    PREFIXED_API(StreamGetId)(stream.stream, &stream_id);

    ASSERT_GT(stream_id, 0u);
#endif

    ASSERT_GE(stream.device(), 0);

    ASSERT_FALSE(stream.capturing());
}

//! @test Check that @ref tests::cuda::View is default constructible.
TEST(APIWrappers, view_default_constructible)
{
    const View<double> data {};
}

//! @test Check that @ref tests::cuda::View construct does not crash.
TEST(APIWrappers, view_constructor)
{
    const Stream stream;

    const View<double> data(stream, 2<<6);

    stream.fence();
}

//! @test Check that @ref tests::cuda::View::get_host_copy works as expected.
TEST(APIWrappers, view_get_host_copy)
{
    const Stream stream;

    const View<double> data(stream, 2<<6);

    const auto mirror = data.get_host_copy(stream);
    stream.fence();

    ASSERT_EQ(mirror.size(), 2<<6);
}

//! @test Check that @ref tests::cuda::View can be copied, still we don't have double free issues.
TEST(APIWrappers, view_double_free)
{
    const Stream stream;

    const View<double> data(stream, 2<<6);

    [[maybe_unused]]const auto other = data; // NOLINT(performance-unnecessary-copy-initialization)

    stream.fence();
}

//! @test Check that @ref tests::cuda::get_driver works as expected.
TEST(APIWrappers, get_driver)
{
    static_assert(std::same_as<decltype(get_driver<Dummy>()), void(*)(const Dummy)>);
}

//! @test Check that @ref tests::cuda::Graph constructor does not crash.
TEST(APIWrappers, graph_constructor)
{
    const Graph graph;
}

/**
 * @test Check that @ref tests::cuda::Graph can be used as a wrapper around a pre-existing @c cudaGraph_t.
 *
 * This test ensures that when @ref tests::cuda::Graph is built from a pre-existing @c cudaGraph_t, it
 * will not destroy it.
 */
TEST(APIWrappers, graph_wrapper_preexisting)
{
    PREFIXED_API(Graph_t) native_graph = nullptr;

    CHECK_CALL(PREFIXED_API(GraphCreate)(&native_graph, 0));

    {
        const Graph graph(native_graph);
    }

    CHECK_CALL(PREFIXED_API(GraphDestroy)(native_graph));
}

//! @test Check that @ref tests::cuda::GraphExecutable constructor does not crash.
TEST(APIWrappers, graph_exec_constructor)
{
    const Graph graph;
    const GraphExecutable graph_exec(graph);
}

//! @test Check that @ref tests::cuda::GraphExecutable::submit does not crash.
TEST(APIWrappers, graph_exec_submit)
{
    const Stream stream;
    const Graph graph;
    const GraphExecutable graph_exec(graph);
    graph_exec.submit(stream);
    stream.fence();
}

} // namespace tests::cuda
