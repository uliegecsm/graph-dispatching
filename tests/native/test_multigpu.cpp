#include <filesystem>

#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Play with multi-GPU @c Cuda graph
 * ---------------------------------
 *
 * This test shows how a multi-GPU @c Cuda graph can be used.
 *
 * The test can be found in @ref native/test_multigpu.cpp.
 */

namespace tests::cuda
{

//! Simple kernel that spins until the last element of @p data is not zero.
template <typename T>
__global__ void my_long_kernel(const T data)
{
    size_t iter = 0;
    while(atomicMax(&data(data.size - 1), 0) == 0) {++iter;}
    data(data.size - 1) = iter;
}

#define DEVICE_ID_0 0
#define DEVICE_ID_1 1

/**
 * @test Check how a multi-GPU @c Cuda graph can be used.
 *
 * @warning This test works for @c HIP because we are in an edge case that, as of @c HIP 6.3.0, works.
 *          Indeed, given the code at https://github.com/ROCm/clr/blob/7c9c7a6332f69f740f59aaaeb83ad6eeff4598d6/hipamd/src/hip_graph_internal.cpp#L317-L335,
 *          it is clear the executable graph will allocate its pool of streams on the current device.
 *          It means that it is not really implemented for multi-GPU support yet.
 *          In this case, we instantiate the graph on the GPU 1, and following the algorithm used to assign a stream ID
 *          to each node (see https://github.com/ROCm/clr/blob/7c9c7a6332f69f740f59aaaeb83ad6eeff4598d6/hipamd/src/hip_graph_internal.cpp#L201),
 *          there will be a pool of 2 streams. The nodes we want to run on GPU 0 will be associated to the stream ID 0 and the node
 *          we want to run on GPU 1 to stream ID 1.
 *          Finally, we submit the graph on the stream associated to GPU 0 and given
 *          https://github.com/ROCm/clr/blob/7c9c7a6332f69f740f59aaaeb83ad6eeff4598d6/hipamd/src/hip_graph_internal.cpp#L706,
 *          the pool of streams will be updated such that the stream ID 0 of the pool is the stream we created on GPU 0,
 *          and the stream ID 1 of the pool is the one created during instantiation on GPU 1.
 *          Therefore, by chance, this test works though in general, @c HIP graph does not support multi-GPU yet.
 *          See also https://github.com/ROCm/clr/issues/113.
 */
TEST(cuda, multigpu)
{
    using view_t    = View<int>;
    using functor_t = MyFunctor<view_t>;

    constexpr size_t size = 2<<4;

    int num_devs = 0;
    CHECK_CALL(PREFIXED_API(GetDeviceCount)(&num_devs));

    ASSERT_GE(num_devs, 2) << "This test requires at least 2 GPUs.";

    //! Create stream and data on GPU 0.
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_0));
    Stream stream_0;
    view_t data_0(stream_0, size);

    //! Create stream and data on GPU 1.
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_1));
    Stream stream_1;
    view_t data_1(stream_1, size);

    //! The device on which we create the graph does not matter. Change to @c DEVICE_ID_0 to convince yourself.
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_0));
    Graph graph;

    //! Add one node that will run on GPU 0.
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_0));
    functor_t functor_0{.data = data_0};
    GraphNodeKernel work_0(functor_0, size);
    work_0.add(graph, {});

    //! Add one node that will run on GPU 1.
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_1));
    functor_t functor_1{.data = data_1};
    GraphNodeKernel work_1(functor_1, size);
    work_1.add(graph, {});

    //! Add one node that will run on GPU 0 when both previous nodes are done.
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_0));
    functor_t functor_2{.data = data_0};
    GraphNodeKernel work_2(functor_2, size);
    work_2.add(graph, {work_0, work_1});

    //! The graph is ready, dump it for later debug.
    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_multigpu.dot").c_str(), PREFIXED_API(GraphDebugDotFlagsVerbose));

    //! Fence streams to ensure the data is ready.
    stream_0.fence();
    stream_1.fence();

    /// Run some long work on stream of GPU 1.
    /// If the graph schedules the node @c work_1 (that should run on GPU 1) using the stream attached to GPU1, it might hang.
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_1));
    my_long_kernel<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, stream_1.stream>>>(view_t(data_1.size, data_1.buffer));

    /// Instantiate the graph on GPU 1. Doing it on GPU 0 fails, but not sure why.
    /// It should not be impacted by the "long" job running on the stream of GPU 1.
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_1));
    GraphExecutable graph_exec(graph);
    CHECK_CALL(PREFIXED_API(SetDevice)(DEVICE_ID_0));
    graph_exec.submit(stream_0);

    //! Fence both GPUs.
    stream_0.fence();
    stream_1.fence();
}

} // namespace tests::cuda
