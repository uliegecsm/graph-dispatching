#include <filesystem>

#include "gtest/gtest.h"

#include "cusparse.h"

#include "tests/cuda/APIWrappers_def.hpp"
#include "tests/cuda/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Graph capture
 * -------------
 *
 * This test shows how one can combine "regular" graph definition and graph capture, following
 * a modified version of the @c cuSPARSE example at
 * https://github.com/NVIDIA/CUDALibrarySamples/blob/5cd2c16eb63ca861a34faaf099c1a9e80a500d56/cuSPARSE/graph_capture/graph_capture_example.c.
 *
 * The test can be found in @ref cuda/test_graph_capture.cpp.
 */

namespace tests::cuda
{

namespace sparse
{

struct Handle
{
    cusparseHandle_t handle = nullptr;
    Handle()  { CHECK_SPARSE_CALL(cusparseCreate(&handle)); }
    ~Handle() { CHECK_SPARSE_CALL(cusparseDestroy(handle)); }
    void set_stream(const Stream& stream) const { CHECK_SPARSE_CALL(cusparseSetStream(handle, stream.stream)); }
};

struct SparseVectorDescriptor
{
    cusparseSpVecDescr_t descr = nullptr;
    ~SparseVectorDescriptor() { CHECK_SPARSE_CALL(cusparseDestroySpVec(descr)); }
};

struct DenseVectorDescriptor
{
    cusparseDnVecDescr_t descr = nullptr;
    ~DenseVectorDescriptor() { CHECK_SPARSE_CALL(cusparseDestroyDnVec(descr)); }
};

} // namespace sparse

//! @test Use graph capture with @c cuSPARSE. The captured nodes are directly added to the main graph.
TEST(cuda, graph_capture)
{
    using value_t   = double;
    using view_t    = View<value_t>;
    using functor_t = MyFunctor<view_t>;

    constexpr size_t size = 5;

    Stream stream;

    //! Create data on device.
    view_t    dense_val (stream, std::array{0., 1., 2., 3., 4.});
    view_t    sparse_val(stream, std::array{1., 2., 3., 4., 5.});
    View<int> sparse_ind(stream, std::array{0 , 1 , 2 , 3 , 4 });

    ASSERT_EQ(dense_val .size, size);
    ASSERT_EQ(sparse_val.size, size);
    ASSERT_EQ(sparse_ind.size, size);

    //! Placeholder for the result.
    value_t result = 0.;

    //! Create a @c cuSPARSE handle.
    sparse::Handle handle;

    //! Describe vectors to @c cuSPARSE.
    sparse::SparseVectorDescriptor sparse_descr;
    sparse::DenseVectorDescriptor  dense_descr;

    CHECK_SPARSE_CALL(cusparseCreateSpVec(
        &sparse_descr.descr,
        size, size, sparse_ind.buffer, sparse_val.buffer,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
    ));
    CHECK_SPARSE_CALL(cusparseCreateDnVec(
        &dense_descr.descr,
        size, dense_val.buffer,
        CUDA_R_64F
    ));

    //! Allocate buffer memory for @c cusparseSpVV.
    size_t buffer_size = 0;
    CHECK_SPARSE_CALL(cusparseSpVV_bufferSize(
        handle.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        sparse_descr.descr, dense_descr.descr,
        &result,
        CUDA_R_64F,
        &buffer_size
    ));
    View<void> buffer(stream, buffer_size);

    //! Check buffer size. It's used for the result of @c cusparseSpVV (see below).
    ASSERT_EQ(buffer_size, 8);

    //! Create the graph.
    Graph graph;

    //! Add a node "in a regular way".
    functor_t functor{.data = dense_val};
    GraphNodeKernel node(functor, size);
    node.add(graph);

    //! Enable graph capture, telling the capture logic where to add captured nodes.
    handle.set_stream(stream);

    const std::vector<cudaGraphNode_t> dependencies {node.node};
    CHECK_CALL(PREFIXED_API(StreamBeginCaptureToGraph)(
        stream.stream, graph.graph,
        dependencies.data(), nullptr, dependencies.size(),
        PREFIXED_API(StreamCaptureModeGlobal)
    ));

    //! Check that the stream status is "actively capturing".
    PREFIXED_API(StreamCaptureStatus) stream_capturing = cudaStreamCaptureStatusNone;
    CHECK_CALL(PREFIXED_API(StreamIsCapturing)(stream.stream, &stream_capturing));
    ASSERT_EQ(stream_capturing, cudaStreamCaptureStatusActive);

    //! Add @c cuSPARSE call to the graph.
    CHECK_SPARSE_CALL(cusparseSpVV(
        handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        sparse_descr.descr, dense_descr.descr,
        &result, CUDA_R_64F, buffer.buffer
    ));

    //! Get the "tail node of captured graph branch" before stopping the capture, so we can add nodes later on.
    const cudaGraphNode_t* capture_dependencies_out_nodes = nullptr;
    size_t capture_dependencies_out_count = 0;

    CHECK_CALL(cudaStreamGetCaptureInfo_v3(
        stream.stream, &stream_capturing, nullptr,
        nullptr, &capture_dependencies_out_nodes, nullptr,
        &capture_dependencies_out_count
    ));

    ASSERT_EQ(stream_capturing, cudaStreamCaptureStatusActive);
    ASSERT_EQ(capture_dependencies_out_count, 1);

    //! Stop graph capture.
    CHECK_CALL(PREFIXED_API(StreamEndCapture)(stream.stream, &graph.graph));

    //! Add one last node, that follows the captured nodes.
    GraphNode node_capture_out {.node = capture_dependencies_out_nodes[0] };
    functor_t functor_end{.data = dense_val};
    GraphNodeKernel node_end(functor_end, size);
    node_end.add(graph, {node_capture_out});

    /// Check content of the graph.
    /// Since the result is on host, there will be a @c memcpy node in addition to the kernel nodes.
    graph.print((std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / "test_graph_capture.dot").c_str(), cudaGraphDebugDotFlagsVerbose);

    size_t num_nodes = 0;
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nullptr, &num_nodes)));
    ASSERT_EQ(num_nodes, 4);

    std::array<PREFIXED_API(GraphNode_t), 4> nodes;
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nodes.data(), &num_nodes)));

    PREFIXED_API(GraphNodeType) node_type;

    CHECK_CALL(PREFIXED_API(GraphNodeGetType)(nodes.at(0), &node_type));
    ASSERT_EQ(node_type, cudaGraphNodeTypeKernel);

    CHECK_CALL(PREFIXED_API(GraphNodeGetType)(nodes.at(1), &node_type));
    ASSERT_EQ(node_type, cudaGraphNodeTypeKernel);

    CHECK_CALL(PREFIXED_API(GraphNodeGetType)(nodes.at(2), &node_type));
    ASSERT_EQ(node_type, cudaGraphNodeTypeMemcpy);

    CHECK_CALL(PREFIXED_API(GraphNodeGetType)(nodes.at(3), &node_type));
    ASSERT_EQ(node_type, cudaGraphNodeTypeKernel);

    size_t num_edges = 0;
    CHECK_CALL(PREFIXED_API(GraphGetEdges(graph.graph, nullptr, nullptr, &num_edges)));
    ASSERT_EQ(num_edges, 3);

    std::array<PREFIXED_API(GraphNode_t), 3> edges_from, edges_to;
    CHECK_CALL(PREFIXED_API(GraphGetEdges(graph.graph, edges_from.data(), edges_to.data(), &num_edges)));

    ASSERT_EQ(edges_from.at(0), nodes.at(0)); ASSERT_EQ(edges_to.at(0), nodes.at(1));
    ASSERT_EQ(edges_from.at(1), nodes.at(1)); ASSERT_EQ(edges_to.at(1), nodes.at(2));
    ASSERT_EQ(edges_from.at(2), nodes.at(2)); ASSERT_EQ(edges_to.at(2), nodes.at(3));

    //! Instantiate and submit.
    GraphExecutable graph_exec(graph);

    graph_exec.submit(stream);

    //! Check result (sum of squares of natural numbers).
    ASSERT_EQ(result, size * (size + 1) * (2 * size + 1) / 6);

    //! Check content of the dense vector (2 kernels have atomically add one to each element).
    ASSERT_EQ(dense_val.get_host_copy(stream), (std::vector<value_t>{2., 3., 4., 5., 6.}));

    /// Ensure that the buffer was used to store the result of the dot product on device, and
    /// that the @c memcpy node used it to transfer to the host.
    value_t buffer_value;
    buffer.get_host_copy(stream, &buffer_value);

    ASSERT_EQ(buffer_value, result);

    cudaMemcpy3DParms node_memcpy_params;
    CHECK_CALL(PREFIXED_API(GraphMemcpyNodeGetParams)(nodes.at(2), &node_memcpy_params));

    ASSERT_EQ(node_memcpy_params.dstPtr.ptr, &result);
    ASSERT_EQ(node_memcpy_params.srcPtr.ptr, buffer.buffer);
}

} // namespace tests::cuda
