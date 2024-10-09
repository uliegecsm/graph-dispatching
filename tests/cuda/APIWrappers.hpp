#ifndef GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_HPP
#define GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_HPP

/**
 * @file
 *
 * This file contains wrappers for the @c Cuda API.
 *
 * @note It can easily be used for @c HIP as well, see @ref PREFIXED_API.
 * 
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

//! Check the return code of an API call.
#define CHECK_CALL(call)                                      \
    {                                                         \
        const auto error_code = call;                         \
        if(error_code != PREFIXED_API(Success))               \
        {                                                     \
            printf("Failure of statement %s: %s\n", #call,    \
                   PREFIXED_API(GetErrorString)(error_code)); \
            std::abort();                                     \
        }                                                     \
    }

namespace tests::cuda
{
//! Wrapper for a stream.
struct Stream
{
    PREFIXED_API(Stream_t) stream;

    Stream();

    void fence() const;
};

//! Wrapper for a view over data.
template <typename T>
struct View
{
    size_t size = 0;
    T* buffer = nullptr;
    bool owning = false;

    View(const Stream& stream, const size_t size_);

    //! This is the simplest approach. The original view is the only one that owns and frees.
    View(const View& other) : size(other.size), buffer(other.buffer), owning(false) {}

    //! Get a host copy of the buffer.
    auto get_host_copy(const Stream& stream) const;

    KOKKOS_FUNCTION
    auto& operator()(const unsigned int index) const { return buffer[index]; }

    ~View();
};

//! Wrapper for a graph.
struct Graph
{
    PREFIXED_API(Graph_t) graph;
    bool owning = true;

    Graph();

    Graph(const PREFIXED_API(Graph_t) graph_) : graph(graph_), owning(false) {}

    void print(const char* path, const unsigned int flags);

    ~Graph();
};

//! Wrapper for a graph node.
struct GraphNode
{
    PREFIXED_API(GraphNode_t) node = nullptr;

    auto transform_to_impl(const std::vector<GraphNode>& ancestors = {});
};

//! Wrapper for an executable graph.
struct GraphExecutable
{
    PREFIXED_API(GraphExec_t) graph_exec;

    GraphExecutable(const Graph& graph);

    ~GraphExecutable();

    void submit(const Stream& stream);
};

//! Wrapper for a kernel node.
template <typename Functor>
struct GraphNodeKernel : public GraphNode
{
    PREFIXED_API(KernelNodeParams) params;
    std::vector<void*> inputs;

    GraphNodeKernel(const Functor& functor, const size_t shape);

    void add(const Graph& graph, const std::vector<GraphNode>& ancestors = {});
};

//! Similar to @c Kokkos driver (local memory launch).
template <typename Functor> requires ( ! std::is_pointer_v<Functor> )
__global__ void driver(const Functor functor);

template <typename Functor>
auto get_driver() { return driver<Functor>; }

} // namespace tests::cuda

#endif // GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_HPP
