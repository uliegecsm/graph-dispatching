#include <filesystem>

#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"

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

enum class Mode : std::uint8_t
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
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
        CHECK_SPARSE_CALL(cusparseSpVV_bufferSize(
            handle.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            sparse_descr.descr, dense_descr.descr,
            &result,
            sparse::DataType<value_t>::type,
            &buffer_size
        ));
PRAGMA_DIAGNOSTIC_POP

        //! Check buffer size. It's used for the result of @c cusparseSpVV (see @ref result).
#if CUDART_VERSION >= 13000
        ASSERT_EQ(buffer_size, 15);
#else
        ASSERT_EQ(buffer_size, sizeof(value_t));
#endif
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

        cudaMemcpy3DParms node_memcpy_params {};
        if constexpr (mode == Mode::SINGLE) {
            CHECK_CALL(PREFIXED_API(GraphMemcpyNodeGetParams)(nodes.at(2), &node_memcpy_params));
        } else {
            //! In @ref Mode::SUBGRAPH mode, we need to retrieve the @c memcpy node from the child graph node.
            Graph graph(nullptr);
            CHECK_CALL(PREFIXED_API(GraphChildGraphNodeGetGraph)(nodes.at(1), &graph.graph));

            std::array<PREFIXED_API(GraphNode_t), 2> child_graph_nodes {};
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
        const functor_t functor_end{.data = dense};
        GraphNodeKernel node_end(functor_end, size);
        node_end.add(graph, {captured});

        this->print(graph);

        const auto nodes = this->check_topology<mode>(graph);

        const GraphExecutable graph_exec(graph);

        graph_exec.submit(stream);

        this->check_results();

        this->check_buffer<mode>(nodes);
    }

    //! Call @c cusparseSpVV on @p cu_stream.
    void cusparseSpVV(const Stream& cu_stream)
    {
        handle.set_stream(cu_stream);

PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
        CHECK_SPARSE_CALL(::cusparseSpVV(
            handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            sparse_descr.descr, dense_descr.descr,
            &result, sparse::DataType<value_t>::type,
            buffer.buffer
        ));
PRAGMA_DIAGNOSTIC_POP
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

    std::array<PREFIXED_API(GraphNode_t), 4> nodes {};
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nodes.data(), &num_nodes)));

    EXPECT_NODE_TYPE_EQ(nodes.at(0), Kernel);
    EXPECT_NODE_TYPE_EQ(nodes.at(1), Kernel);
    EXPECT_NODE_TYPE_EQ(nodes.at(2), Memcpy);
    EXPECT_NODE_TYPE_EQ(nodes.at(3), Kernel);

    const size_t num_edges = graph.get_num_edges();
    EXPECT_EQ(num_edges, 3);

    const auto [edges_from, edges_to] = graph.get_edges();

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

    std::array<PREFIXED_API(GraphNode_t), 3> nodes {};
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nodes.data(), &num_nodes)));

    EXPECT_NODE_TYPE_EQ(nodes.at(0), Kernel);
    EXPECT_NODE_TYPE_EQ(nodes.at(1), Graph );
    EXPECT_NODE_TYPE_EQ(nodes.at(2), Kernel);

    const size_t num_edges = graph.get_num_edges();
    EXPECT_EQ(num_edges, 2);

    const auto [edges_from, edges_to] = graph.get_edges();

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

    CHECK_CALL(
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaStreamGetCaptureInfo
#else
    cudaStreamGetCaptureInfo_v3
#endif
    (
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

/**
 * To support graph mode, we need to pass to functions at least:
 *  - node(s) we will be appending to (a.k.a. *predecessors*)
 *  - execution space instance for allocations
 *
 * In @c Kokkos, we would not need to pass a @ref graph since adding a node is done through another node. But here we don't have
 * such mechanisms so we also pass the graph.
 */
struct StatefulGraphNode
{
    GraphNode node;
    Stream    stream;
    Graph     graph;
};

//! Retrieve the @ref StatefulGraphNode::stream.
auto& get_stream(const StatefulGraphNode& node) { return node.stream; }

//! @overload
auto& get_stream(const Stream& stream) { return stream; }

//! Capture if @p exec is a @ref StatefulGraphNode.
template <typename Exec, typename Closure>
void capture(const Exec& exec, Closure&& closure)
{
    if constexpr (std::same_as<std::remove_cvref_t<Exec>, StatefulGraphNode>) {
        CHECK_CALL(PREFIXED_API(StreamBeginCapture)(get_stream(exec).stream, PREFIXED_API(StreamCaptureModeGlobal)));
    }

    std::forward<Closure>(closure)(get_stream(exec));

    if constexpr (std::same_as<std::remove_cvref_t<Exec>, StatefulGraphNode>)
    {
        Graph library(nullptr);

        CHECK_CALL(PREFIXED_API(StreamEndCapture)(get_stream(exec).stream, &library.graph));

        [[maybe_unused]] const auto library_as_node = library.add(exec.graph, std::vector{exec.node});
    }
}

namespace sparse
{

/**
 * @brief Dot product of a @ref dense view with a @ref sparse view.
 *
 * The member @ref buffer is a "state" whose lifetime must be extended in graph mode.
 */
template <typename T>
struct DotImpl
{
    using buffer_t = ::tests::cuda::View<void>;

    ::tests::cuda::View<T> dense;
    ::tests::cuda::sparse::View<T> sparse;

    /// If used in graph mode, the buffer needs to stay alive as long as the graph can be submitted.
    /// @note We need proper reference counting when this class gets moved around. As @ref ::tests::cuda::View cannot do that,
    /// we simply embed it into a shared pointer.
    std::shared_ptr<buffer_t> buffer = nullptr;

    template <typename Exec, typename Result>
    void apply(const Exec& exec, Result& result)
    {
        sparse::Handle handle;

        sparse::DenseVectorDescriptor <T> dense_descr (this->dense);
        sparse::SparseVectorDescriptor<T> sparse_descr(this->sparse);

        size_t buffer_size = 0;

PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
        CHECK_SPARSE_CALL(cusparseSpVV_bufferSize(
            handle.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            sparse_descr.descr, dense_descr.descr,
            &result,
            sparse::DataType<T>::type,
            &buffer_size
        ));
PRAGMA_DIAGNOSTIC_POP

        buffer = std::make_shared<buffer_t>(get_stream(exec), buffer_size);

        /// @warning We need to start graph capture as late as possible. Indeed, with graph capture enabled,
        ///          we could not allocate @ref buffer because it's prohibited by @c Cuda to allocate while capturing.
        ///          We could use the "relaxed" capture mode to allocate into another stream, but it means creating a new stream here.
        ///          It's also quite "hard" to make @ref buffer a graph memory node without a lot of changes to the structure of the code.
        ///
        /// @note The lambda receives a @ref tests::cuda::Stream because its call operator should be written as if we never knew
        ///       about the graph thingy. Also, the lambda should be viewed as what's defining the workload, but it should not
        ///       capture on which stream it will be executed.
        capture(
            exec,
            [&, buffer = buffer](const Stream& stream) {
                handle.set_stream(stream);

                printf("> Calling 'cusparseSpVV' with stream capturing(=%s).\n", stream.capturing() ? "true" : "false");

PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
                CHECK_SPARSE_CALL(::cusparseSpVV(
                    handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    sparse_descr.descr, dense_descr.descr,
                    &result, sparse::DataType<T>::type,
                    buffer->buffer
                ));
PRAGMA_DIAGNOSTIC_POP
            }
        );
    }
};

/**
 * @brief Perform the dot product of @p dense and @p sparse using @c cusparseSpVV and place result in @p result.
 *
 * This function needs to create "states":
 *  - descriptors of @p dense and @p sparse
 *  - allocate a buffer for the @p result
 *
 * Therefore, if this function is captured in a graph, these states should have their lifetime extended to the duration
 * of the graph. Note that in this case, it seems that we don't need to extend the lifetime of the @c cuSPARSE descriptors, *i.e.*
 * they can be deleted even before we instantiate the graph (see @ref DotImpl::apply).
 *
 * With @c Kokkos::Graph, we'd need to graft these states to the @c Kokkos node, since the underlying @c Cuda node cannot have additional
 * "state" members.
 *
 * At first, one could argue that this is the purpose of @c cudaUserObject_t. But it's clearly not a portable solution.
 * Moreover, there can't be any @c Cuda API call in the destructor of such a @c cudaUserObject_t object, thereby restricting what can be done in it.
 * For instance, though we need to extend the lifetime of @ref DotImpl::buffer, we cannot use @c cudaFree when the destructor of @ref DotImpl
 * is called by the @c cudaUserObject_t.
 *
 * For now, we simply return a "handle" that the user must store. In the long term, this should be grafted to the @c Kokkos graph, and @c Kokkos graph
 * would first destroy its @c Cuda members before destroying all the "grafted states".
 *
 * See "User objects" in:
 *  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-user-objects
 *  - https://developer.nvidia.com/blog/exploring-the-new-features-of-cuda-11-3/
 */
template <typename Exec, typename T, typename Result> requires (std::is_scalar_v<Result>)
[[nodiscard]] auto dot(const Exec& exec, const ::tests::cuda::View<T>& dense, const ::tests::cuda::sparse::View<T>& sparse, Result& result)
{
    DotImpl<T> impl{.dense = dense, .sparse = sparse};

    impl.apply(exec, result);

    return impl;
};

} // namespace sparse

class GraphCaptureWithProlongedState : public ::testing::Test
{
public:
    using value_t       = double;
    using dense_view_t  = ::tests::cuda::View<value_t>;
    using sparse_view_t = ::tests::cuda::sparse::View<value_t>;
    using functor_t     = MyFunctor<dense_view_t>;

    static constexpr size_t size = 5;

public:
    void SetUp() override
    {
        this->dense  = dense_view_t(stream, std::array{1., 2., 3., 4., 5.});
        this->sparse = sparse_view_t {
            .indices = View<int>   (stream, std::array{0 , 1 , 2 , 3 , 4 }),
            .values  = dense_view_t(stream, std::array{1., 2., 3., 4., 5.})
        };
    }

protected:
    Stream        stream;
    dense_view_t  dense;
    sparse_view_t sparse;
};

//! @test Check that @ref sparse::dot works without graph capture.
TEST_F(GraphCaptureWithProlongedState, dot_standalone)
{
    double result = 0;

    { [[maybe_unused]] const auto tmp = dot(stream, dense, sparse, result); }

    stream.fence();

    ASSERT_EQ(result, size * (size + 1) * (2 * size + 1) / 6);
}

/**
 * @test Similar to @ref GraphCaptureTest_as_a_subgraph_Test, but the captured nodes now embed the necessary state
 *       variables (those of @c cuSPARSE).
 *
 * @note It sort of mimics what we could find in @c KokkosKernels for instance.
 */
TEST_F(GraphCaptureWithProlongedState, dot_captured_in_graph)
{
    double result = 0.;

    //! Create the graph, with a node that increments by one all elements of @ref dense.
    const Graph graph;

    const functor_t functor{.data = dense};
    GraphNodeKernel node(functor, size);
    node.add(graph);

    //! Add @ref sparse::dot to the graph.
    [[maybe_unused]] const auto tmp = dot(
        StatefulGraphNode{.node = node, .stream = Stream(stream.stream), .graph = Graph(graph.graph)},
        dense, sparse, result
    );

    //! Check nodes.
    size_t num_nodes = 2;
    std::array<PREFIXED_API(GraphNode_t), 2> nodes {};
    CHECK_CALL(PREFIXED_API(GraphGetNodes(graph.graph, nodes.data(), &num_nodes)));

    EXPECT_NODE_TYPE_EQ(nodes.at(0), Kernel);
    EXPECT_NODE_TYPE_EQ(nodes.at(1), Graph);

    //! Create the executable graph.
    const GraphExecutable graph_exec(graph);

    //! Submit many times.
    for(unsigned short int irep = 0; irep < 5; ++irep)
    {
        graph_exec.submit(stream);
        stream.fence();

        ASSERT_EQ(result, [&](){
            double sum = 0.;
            for(size_t ielem = 0; ielem < size; ++ielem) {
                sum += (ielem + 1 + irep + 1) * (ielem + 1);
            }
            return sum;
        }());
    }
}

} // namespace tests::cuda
