#include <filesystem>

#include "gtest/gtest.h"

#include "tests/native/APIWrappers_def.hpp"
#include "tests/native/APIWrappers_sparse.hpp"
#include "tests/native/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Graph capture
 * -------------
 *
 * This group of tests show how one can combine "regular" graph definition and graph capture, following
 * a modified version of the @c cuSPARSE example at
 * https://github.com/NVIDIA/CUDALibrarySamples/blob/5cd2c16eb63ca861a34faaf099c1a9e80a500d56/cuSPARSE/graph_capture/graph_capture_example.c.
 *
 * In @ref tests::cuda::Mode::SINGLE mode, the captured nodes are embedded directly into the main application graph.
 * In @ref tests::cuda::Mode::SUBGRAPH mode, the captured nodes are added to a separated graph, that is later on grafted
 * to the main application graph.
 *
 * Both modes are valid, but @ref tests::cuda::Mode::SINGLE requires a bit more plumbing to connect the main application
 * to the captured nodes.
 *
 * The tests can be found in @ref native/test_graph_capture.cpp.
 */

namespace tests::cuda
{

//! Check that @p __node__ is of type @p __type__.
#define EXPECT_NODE_TYPE_EQ(__node__, __type__)                           \
    {                                                                     \
        PREFIXED_API(GraphNodeType) node_type;                            \
        CHECK_CALL(PREFIXED_API(GraphNodeGetType)(__node__, &node_type)); \
        EXPECT_EQ(node_type, cudaGraphNodeType##__type__);                \
    }

enum class Mode
{
    SINGLE,  //!< Embed the external library call directly into the main application graph.
    SUBGRAPH //!< Embed the external library call as a subgraph into the main application graph.
};

class GraphCaptureTest : public ::testing::Test
{
public:
    using value_t       = double;
    using dense_view_t  = View<value_t>;
    using sparse_view_t = sparse::View<value_t>;
    using functor_t     = MyFunctor<dense_view_t>;

    using buffer_t = View<void>;

    static constexpr size_t size = 5;

public:
    void SetUp() override
    {
        //! Create data on device.
        this->dense  = dense_view_t(stream, std::array{0., 1., 2., 3., 4.});
        this->sparse = sparse_view_t {
            .indices = View<int>   (stream, std::array{0 , 1 , 2 , 3 , 4 }),
            .values  = dense_view_t(stream, std::array{1., 2., 3., 4., 5.})
        };

        ASSERT_EQ(dense         .size, size);
        ASSERT_EQ(sparse.indices.size, size);
        ASSERT_EQ(sparse.values .size, size);

        //! Initialize @c cuSPARSE descriptors.
        sparse_descr = sparse::SparseVectorDescriptor<value_t>(sparse);
        dense_descr  = sparse::DenseVectorDescriptor <value_t>(dense);

        //! Allocate buffer memory for @c cusparseSpVV.
        size_t buffer_size = 0;
        CHECK_SPARSE_CALL(cusparseSpVV_bufferSize(
            handle.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            sparse_descr.descr, dense_descr.descr,
            &result,
            sparse::DataType<value_t>::type,
            &buffer_size
        ));

        //! Check buffer size. It's used for the result of @c cusparseSpVV (see @ref result).
        ASSERT_EQ(buffer_size, sizeof(value_t));
        this->buffer = buffer_t(stream, buffer_size);
    }

    //! Debug print @p graph.
    void print(const Graph& graph) const
    {
        const std::string test_name = std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name())
                                                + '_'
                                                + ::testing::UnitTest::GetInstance()->current_test_info()->name();
        const auto        file_name = std::filesystem::path(CMAKE_CURRENT_BINARY_DIR) / (test_name + ".dot");

        printf("> Writing graph as DOT in %s.\n", file_name.c_str());

        graph.print(file_name.c_str(), cudaGraphDebugDotFlagsVerbose);
    }

    /// Check content of the graph (its topology).
    /// Since the @ref result is on host, there will be a @c memcpy
    /// node (added by @c cusparseSpVV) in addition to the kernel nodes.
    template <Mode mode>
    auto check_topology(const Graph& graph) const;

    /// Check value of @ref result (sum of squares of natural numbers).
    /// Check content of the dense vector @ref dense (2 kernels have atomically add one to each element).
    void check_results() const
    {
        ASSERT_EQ(result, size * (size + 1) * (2 * size + 1) / 6);

        ASSERT_EQ(dense.get_host_copy(stream), (std::vector<value_t>{2., 3., 4., 5., 6.}));
    }

    /// Ensure that the buffer was used to store the result of the dot product on device, and
    /// that the @c memcpy node used it to transfer to the host.
    template <Mode mode, typename T>
    void check_buffer(const T& nodes) const
    {
        value_t buffer_value;
        buffer.get_host_copy(stream, &buffer_value);

        ASSERT_EQ(buffer_value, result);

        cudaMemcpy3DParms node_memcpy_params;
        if constexpr (mode == Mode::SINGLE) {
            CHECK_CALL(PREFIXED_API(GraphMemcpyNodeGetParams)(nodes.at(2), &node_memcpy_params));
        } else {
            //! In @ref Mode::SUBGRAPH mode, we need to retrieve the @c memcpy node from the child graph node.
            Graph graph(nullptr);
            CHECK_CALL(PREFIXED_API(GraphChildGraphNodeGetGraph)(nodes.at(1), &graph.graph));

            std::array<PREFIXED_API(GraphNode_t), 2> child_graph_nodes;
            size_t num_nodes = 2;
            CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, child_graph_nodes.data(), &num_nodes)));

            CHECK_CALL(PREFIXED_API(GraphMemcpyNodeGetParams)(child_graph_nodes.at(1), &node_memcpy_params));
        }

        ASSERT_EQ(node_memcpy_params.dstPtr.ptr, &result);
        ASSERT_EQ(node_memcpy_params.srcPtr.ptr, buffer.buffer);
    }

    //! Create the graph and its first node.
    std::tuple<Graph, functor_t, GraphNodeKernel<functor_t>> create_graph() const
    {
        Graph graph;

        functor_t functor{.data = dense};
        GraphNodeKernel node(functor, size);
        node.add(graph);

        return {std::move(graph), std::move(functor), std::move(node)};
    }

    /// Add one last node, that follows the captured nodes.
    /// Once the graph is defined, instantiate it, submit it and check everything.
    template <Mode mode>
    void run(const Graph& graph, const GraphNode& captured) const
    {
        functor_t functor_end{.data = dense};
        GraphNodeKernel node_end(functor_end, size);
        node_end.add(graph, {captured});

        this->print(graph);

        const auto nodes = this->check_topology<mode>(graph);

        GraphExecutable graph_exec(graph);

        graph_exec.submit(stream);

        this->check_results();

        this->check_buffer<mode>(nodes);
    }

    //! Call @c cusparseSpVV on @p stream.
    void cusparseSpVV(const Stream& stream)
    {
        handle.set_stream(stream);

        CHECK_SPARSE_CALL(::cusparseSpVV(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            sparse_descr.descr, dense_descr.descr,
            &result, sparse::DataType<value_t>::type,
            buffer.buffer
        ));
    }

protected:
    Stream stream;

    sparse::Handle handle; //!< @c cuSPARSE handle.

    dense_view_t  dense;  //!< Dense view.
    sparse_view_t sparse; //!< Sparse view.

    sparse::SparseVectorDescriptor<value_t> sparse_descr;
    sparse::DenseVectorDescriptor <value_t> dense_descr;

    buffer_t buffer; //!< Buffer that will be used by @c cusparseSpVV.

    value_t result = 0.; //!< Placeholder for the result of @c cusparseSpVV.
};

template <>
auto GraphCaptureTest::check_topology<Mode::SINGLE>(const Graph& graph) const
{
    size_t num_nodes = 0;
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nullptr, &num_nodes)));
    EXPECT_EQ(num_nodes, 4);

    std::array<PREFIXED_API(GraphNode_t), 4> nodes;
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nodes.data(), &num_nodes)));

    EXPECT_NODE_TYPE_EQ(nodes.at(0), Kernel);
    EXPECT_NODE_TYPE_EQ(nodes.at(1), Kernel);
    EXPECT_NODE_TYPE_EQ(nodes.at(2), Memcpy);
    EXPECT_NODE_TYPE_EQ(nodes.at(3), Kernel);

    size_t num_edges = 0;
    CHECK_CALL(PREFIXED_API(GraphGetEdges(graph.graph, nullptr, nullptr, &num_edges)));
    EXPECT_EQ(num_edges, 3);

    std::array<PREFIXED_API(GraphNode_t), 3> edges_from, edges_to;
    CHECK_CALL(PREFIXED_API(GraphGetEdges(graph.graph, edges_from.data(), edges_to.data(), &num_edges)));

    EXPECT_EQ(edges_from.at(0), nodes.at(0)); EXPECT_EQ(edges_to.at(0), nodes.at(1));
    EXPECT_EQ(edges_from.at(1), nodes.at(1)); EXPECT_EQ(edges_to.at(1), nodes.at(2));
    EXPECT_EQ(edges_from.at(2), nodes.at(2)); EXPECT_EQ(edges_to.at(2), nodes.at(3));

    return nodes;
}

template <>
auto GraphCaptureTest::check_topology<Mode::SUBGRAPH>(const Graph& graph) const
{
    size_t num_nodes = 0;
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nullptr, &num_nodes)));
    EXPECT_EQ(num_nodes, 3);

    std::array<PREFIXED_API(GraphNode_t), 3> nodes;
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nodes.data(), &num_nodes)));

    EXPECT_NODE_TYPE_EQ(nodes.at(0), Kernel);
    EXPECT_NODE_TYPE_EQ(nodes.at(1), Graph );
    EXPECT_NODE_TYPE_EQ(nodes.at(2), Kernel);

    size_t num_edges = 0;
    CHECK_CALL(PREFIXED_API(GraphGetEdges(graph.graph, nullptr, nullptr, &num_edges)));
    EXPECT_EQ(num_edges, 2);

    std::array<PREFIXED_API(GraphNode_t), 2> edges_from, edges_to;
    CHECK_CALL(PREFIXED_API(GraphGetEdges(graph.graph, edges_from.data(), edges_to.data(), &num_edges)));

    EXPECT_EQ(edges_from.at(0), nodes.at(0)); EXPECT_EQ(edges_to.at(0), nodes.at(1));
    EXPECT_EQ(edges_from.at(1), nodes.at(1)); EXPECT_EQ(edges_to.at(1), nodes.at(2));

    return nodes;
}

//! @test Use graph capture with @c cuSPARSE. The captured nodes are directly added to the main graph.
TEST_F(GraphCaptureTest, into_main_graph_directly)
{
    auto [graph, functor, node] = this->create_graph();

    const std::vector<cudaGraphNode_t> dependencies {node.node};
    CHECK_CALL(PREFIXED_API(StreamBeginCaptureToGraph)(
        stream.stream, graph.graph,
        dependencies.data(), nullptr, dependencies.size(),
        PREFIXED_API(StreamCaptureModeGlobal)
    ));

    ASSERT_TRUE(stream.capturing());

    this->cusparseSpVV(stream);

    //! Get the "tail node of captured graph branch" before stopping the capture, so we can add nodes later on.
    const cudaGraphNode_t* capture_dependencies_out_nodes = nullptr;
    size_t capture_dependencies_out_count = 0;

    PREFIXED_API(StreamCaptureStatus) stream_capturing = cudaStreamCaptureStatusNone;

    CHECK_CALL(cudaStreamGetCaptureInfo_v3(
        stream.stream, &stream_capturing, nullptr,
        nullptr, &capture_dependencies_out_nodes, nullptr,
        &capture_dependencies_out_count
    ));

    ASSERT_EQ(stream_capturing, cudaStreamCaptureStatusActive);
    ASSERT_EQ(capture_dependencies_out_count, 1);

    CHECK_CALL(PREFIXED_API(StreamEndCapture)(stream.stream, &graph.graph));

    this->run<Mode::SINGLE>(graph, GraphNode {.node = capture_dependencies_out_nodes[0]});
}

/**
 * @test Similar to @ref GraphCaptureTest_into_main_graph_directly_Test, but the captured nodes are added to a
 *       subgraph that will be grafted to the main graph.
 */
TEST_F(GraphCaptureTest, as_a_subgraph)
{
    auto [graph, functor, node] = this->create_graph();

    CHECK_CALL(PREFIXED_API(StreamBeginCapture)(stream.stream, PREFIXED_API(StreamCaptureModeGlobal)));

    ASSERT_TRUE(stream.capturing());

    this->cusparseSpVV(stream);

    Graph library(nullptr);

    CHECK_CALL(PREFIXED_API(StreamEndCapture)(stream.stream, &library.graph));

    const auto library_as_node = library.add(graph, {node});

    this->run<Mode::SUBGRAPH>(graph, library_as_node);
}

} // namespace tests::cuda
