#ifndef GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_HPP

#include <span>
#include <vector>

/**
 * @file
 *
 * This file contains wrappers for the @c Cuda API.
 *
 * @note It can easily be used for @c HIP as well, see @ref PREFIXED_API.
 */

//! Same behavior as @c Kokkos conterpart.
#define KOKKOS_FUNCTION __host__ __device__

#if defined(GRAPH_DISPATCHING_ENABLE_CUDA)
    #define PREFIXED_API(what) cuda ##what
    #include <cuda.h>
#elif defined(GRAPH_DISPATCHING_ENABLE_HIP)
    #define PREFIXED_API(what) hip ##what
    #include <hip/hip_runtime.h>
#else
    //! Use this macro to prefix backend-specific names.
    #define PREFIXED_API
    #error "You must enable HIP or Cuda."
#endif

#define CHECK_CALL_IMPL(call, success, get_error)               \
    {                                                           \
        const auto error_code = call;                           \
        if(error_code != success)                               \
        {                                                       \
            printf("%s:%d: failure of statement %s: %s (%d)\n", \
                __FILE__, __LINE__,                             \
                #call,                                          \
                get_error(error_code), error_code);             \
            std::abort();                                       \
        }                                                       \
    }

//! Check the return code of an API call.
#define CHECK_CALL(call) CHECK_CALL_IMPL(call, PREFIXED_API(Success), PREFIXED_API(GetErrorString))

namespace tests::cuda
{
//! Wrapper for a stream.
struct Stream
{
    PREFIXED_API(Stream_t) stream = nullptr;
    bool owning = true;

    Stream();

    explicit Stream(const PREFIXED_API(Stream_t) stream_) : stream(stream_), owning(false) {}

    ~Stream();

    void fence() const;

    bool capturing() const;

    //! Retrieve device ID.
    int device() const;
};

//! Wrapper for a view over data.
template <typename T>
struct View
{
    size_t size = 0;
    T* buffer = nullptr;
    bool owning = false;

    View() = default;

    View(const Stream& stream, const size_t size_);

    //! Similar to @c Kokkos unmanaged view.
    View(const size_t size_, T* buffer_) : size(size_), buffer(buffer_), owning(false) {}

    //! Move-assignment operator. @note We need to set the @p other to non-owning to ensure it won't free the @ref buffer.
    View& operator=(View&& other);

    View(View&& other);

    //! This is the simplest approach. The original view is the only one that owns and frees.
    View(const View& other) : size(other.size), buffer(other.buffer), owning(false) {}

    //! Build a device view from a @c std::vector.
    View(const Stream& stream, const std::span<const T>& values) requires ( ! std::same_as<T, void> );

    //! Get a host copy of the buffer.
    std::vector<T> get_host_copy(const Stream& stream) const requires ( ! std::same_as<T, void> );

    //! @overload
    void get_host_copy(const Stream& stream, T* ptr) const;

    KOKKOS_FUNCTION
    auto& operator()(const unsigned int index) const { return buffer[index]; }

    ~View();
};

//! Wrapper for a graph node.
struct GraphNode
{
    PREFIXED_API(GraphNode_t) node = nullptr;

    static std::vector<PREFIXED_API(GraphNode_t)> transform_to_impl(const std::vector<GraphNode>& ancestors = {});
};

//! Wrapper for a graph.
struct Graph
{
    PREFIXED_API(Graph_t) graph = nullptr;
    bool owning = true;

    Graph();

    explicit Graph(const PREFIXED_API(Graph_t) graph_) : graph(graph_), owning(false) {}

    //! Move constructor. @note We need to set the @p other to non-owning to ensure it won't free the @ref graph.
    Graph(Graph&& other) : graph(other.graph), owning(other.owning) { if(other.owning) other.owning = false; }

    void print(const char* path, const unsigned int flags) const;

    //! Add this graph as a subgraph of @c other.
    [[nodiscard]] GraphNode add(const Graph& other, const std::vector<GraphNode>& ancestors = {}) const;

    //! Get the number of edges in the graph.
    size_t get_num_edges() const;

    std::tuple<std::vector<PREFIXED_API(GraphNode_t)>, std::vector<PREFIXED_API(GraphNode_t)>>
    get_edges() const;

    ~Graph();
};

//! Wrapper for an executable graph.
struct GraphExecutable
{
    PREFIXED_API(GraphExec_t) graph_exec = nullptr;

    explicit GraphExecutable(const Graph& graph);

    ~GraphExecutable();

    //! Enable/disable @p node.
    void set_enabled(const GraphNode& node, const bool is_enabled) const;

    void submit(const Stream& stream) const;
};

//! Wrapper for a kernel node.
template <typename Functor>
struct GraphNodeKernel : public GraphNode
{
    PREFIXED_API(KernelNodeParams) params;
    std::vector<void*> inputs;

    //! @warning The user is responsible for keeping the @c functor alive.
    GraphNodeKernel(const Functor& functor, const size_t shape);

    void add(const Graph& graph, const std::vector<GraphNode>& ancestors = {});
};

#if defined(GRAPH_DISPATCHING_ENABLE_CUDA) || defined(DOXYGEN)
/**
 * @brief Wrapper for a conditional if node.
 *
 * References:
 *  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#conditional-if-nodes
 */
struct GraphNodeConditionalIf : public GraphNode
{
    PREFIXED_API(GraphNodeParams) params = {};

    explicit GraphNodeConditionalIf(cudaGraphConditionalHandle handle);

    void add(const Graph& graph, const std::vector<GraphNode>& ancestors = {});

    auto get() const { return Graph{params.conditional.phGraph_out[0]}; }
};

/**
 * @brief Wrapper for a conditional while node.
 *
 * According to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#conditional-graph-nodes, the condition
 * is evaluated on device. However, it is not clear if the same restrictions as the device graphs apply. As a matter of
 * fact, @c nsys will not show the nodes embedded in the @c while subgraph, pointing to it being indeed a device graph launch.
 *
 * References:
 *  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#conditional-while-nodes
 */
struct GraphNodeConditionalWhile : public GraphNode
{
    PREFIXED_API(GraphNodeParams) params = {};

    explicit GraphNodeConditionalWhile(cudaGraphConditionalHandle handle);

    void add(const Graph& graph, const std::vector<GraphNode>& ancestors = {});

    auto get() const { return Graph{params.conditional.phGraph_out[0]}; }
};
#endif

/**
 * @brief Wrapper for a host node.
 *
 * References:
 *  - https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaHostNodeParams.html#structcudaHostNodeParams
 */
template <typename Functor>
struct GraphNodeHost : public GraphNode
{
    //! Node parameters as needed by the backend.
    PREFIXED_API(HostNodeParams) params = {};

    /// User data whose sole purpose is to store the functor to be used
    /// and pass it to the host callback.
    struct NodeHostCallbackData {
        Functor functor {};
    };

    NodeHostCallbackData data {};

    //! Since @p functor will be stored in @ref data, allow move semantics.
    template <typename T> requires std::same_as<std::remove_cvref_t<T>, Functor>
    explicit GraphNodeHost(T&& functor);

    void add(const Graph& graph, const std::vector<GraphNode>& ancestors = {});

    //! Callback that will be called by the backend during graph execution.
    static void driver(void* data);
};

/**
 * @brief Memory copy node.
 *
 * @note This node implements @c cudaGraphAddMemcpyNode1D.
 */
template <typename T = void>
struct GraphNodeMemcpy : public GraphNode
{
    void*  src;
    void*  dst;
    size_t size;
    PREFIXED_API(MemcpyKind) kind;

    GraphNodeMemcpy(void* const src, void* const dst, const size_t size, const PREFIXED_API(MemcpyKind) kind = PREFIXED_API(MemcpyDeviceToDevice));

    void add(const Graph& graph, const std::vector<GraphNode>& ancestors = {});
};

//! Memory allocation node.
template <typename T>
struct GraphNodeMemoryAllocation : public GraphNode
{
    PREFIXED_API(MemAllocNodeParams) params {};

    //! Address of the node memory allocation (with graph semantics for validity).
    T* ptr = nullptr;

    //! The @p stream is used to identify the resident device.
    GraphNodeMemoryAllocation(const size_t size, const Stream& stream);

    void add(const Graph& graph, const std::vector<GraphNode>& ancestors = {});
};

//! Memory deallocation node.
struct GraphNodeMemoryFree : public GraphNode
{
    void* ptr = nullptr;

    template <typename T>
    explicit GraphNodeMemoryFree(T* ptr_) : ptr(ptr_) {}

    void add(const Graph& graph, const std::vector<GraphNode>& ancestors = {});
};

//! Similar to @c Kokkos driver (local memory launch).
template <typename Functor> requires ( ! std::is_pointer_v<Functor> )
__global__ void driver(const Functor functor); // NOLINT(performance-unnecessary-value-param)

template <typename Functor>
auto get_driver() { return driver<Functor>; }

} // namespace tests::cuda

#endif // GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_HPP
