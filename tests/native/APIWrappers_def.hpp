#ifndef GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_DEF_HPP
#define GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_DEF_HPP

#include "tests/native/APIWrappers.hpp"

namespace tests::cuda
{
Stream::Stream() {
    CHECK_CALL(PREFIXED_API(StreamCreateWithFlags)(&stream, PREFIXED_API(StreamNonBlocking)));
}

Stream::~Stream() {
    if(owning) CHECK_CALL(PREFIXED_API(StreamDestroy)(stream));
}

void Stream::fence() const {
    CHECK_CALL(PREFIXED_API(StreamSynchronize)(stream));
}

bool Stream::capturing() const
{
    PREFIXED_API(StreamCaptureStatus) status = PREFIXED_API(StreamCaptureStatusNone);
    CHECK_CALL(PREFIXED_API(StreamIsCapturing)(stream, &status));
    return status == PREFIXED_API(StreamCaptureStatusActive);
}

int Stream::device() const
{
#if defined(GRAPH_DISPATCHING_ENABLE_CUDA)
    CUcontext context;
    CUdevice  device_id = -1;

    CHECK_CALL(cudaError_t(cuStreamGetCtx(stream, &context)));
    CHECK_CALL(cudaError_t(cuCtxPushCurrent(context)));
    CHECK_CALL(cudaError_t(cuCtxGetDevice(&device_id)));

    return device_id;
#elif defined(GRAPH_DISPATCHING_ENABLE_HIP)
    hipDevice_t device_id = -1;
    CHECK_CALL(hipStreamGetDevice(stream, &device_id));
    return device_id;
#else
    std::abort();
#endif
}

//! If the type is @c void, its "size" is considered equal to 1.
template <typename T>
struct BufferSize
{
    static auto get(const size_t size) requires std::same_as<T, void> { return size; }
    static auto get(const size_t size)                                { return size * sizeof(T); }
};

template <typename T>
View<T>::View(const Stream& stream, const size_t size_) : size(size_), owning(true)
{
    const auto raw_size = BufferSize<T>::get(size);
    printf("> Allocating data of raw size %zu bytes, memset it to 0.\n", raw_size);
    CHECK_CALL(PREFIXED_API(MallocAsync)(&buffer,    raw_size, stream.stream));
    CHECK_CALL(PREFIXED_API(MemsetAsync)( buffer, 0, raw_size, stream.stream));
}

template <typename T>
View<T>& View<T>::operator=(View<T>&& other)
{
    this->size   = other.size;
    this->buffer = other.buffer;
    this->owning = other.owning;
    if(other.owning) other.owning = false;
    return *this;
}

template <typename T>
std::vector<T> View<T>::get_host_copy(const Stream& stream) const requires ( ! std::same_as<T, void> )
{
    std::vector<T> host_copy(size);
    CHECK_CALL(PREFIXED_API(MemcpyAsync)(host_copy.data(), buffer, BufferSize<T>::get(size), PREFIXED_API(MemcpyDeviceToHost), stream.stream));
    return host_copy;
}

template <typename T>
void View<T>::get_host_copy(const Stream& stream, T* ptr) const
{
    CHECK_CALL(PREFIXED_API(MemcpyAsync)(ptr, buffer, BufferSize<T>::get(size), PREFIXED_API(MemcpyDeviceToHost), stream.stream));
}

template <typename T>
View<T>::~View()
{
    if(owning)
    {
        printf("> Deallocation of buffer of raw size %zu bytes at %p.\n", BufferSize<T>::get(size), buffer);
        CHECK_CALL(PREFIXED_API(Free)(buffer));
    }
}

template <typename T>
View<T>::View(const Stream& stream, const std::span<const T>& values) requires ( ! std::same_as<T, void> ) : size(values.size()), owning(true)
{
    const auto raw_size = BufferSize<T>::get(size);
    printf("> Allocating data of raw size %zu bytes, memset it to values of a std::vector at %p.\n", raw_size, values.data());
    CHECK_CALL(PREFIXED_API(MallocAsync)(&buffer, raw_size, stream.stream));
    CHECK_CALL(PREFIXED_API(MemcpyAsync)(buffer, values.data(), raw_size, PREFIXED_API(MemcpyHostToDevice), stream.stream));
}

Graph::Graph()
{
    printf("> Creating a graph.\n");
    CHECK_CALL(PREFIXED_API(GraphCreate)(&graph, 0));
}

Graph::~Graph()
{
    if(owning)
    {
        printf("> Destroying graph at %p.\n", graph);
        CHECK_CALL(PREFIXED_API(GraphDestroy)(graph));
    }
}

void Graph::print(const char* path, const unsigned int flags) const
{
    printf("> Exporting graph %p to %s with flags %u.\n", graph, path, flags);
    CHECK_CALL(PREFIXED_API(GraphDebugDotPrint)(graph, path, flags));
}

GraphNode Graph::add(const Graph& other, const std::vector<GraphNode>& ancestors) const
{
    printf("> Adding graph %p as child graph node to graph %p with %zu ancestors.\n", other.graph, this->graph, ancestors.size());

    GraphNode child;

    const auto ancestors_impl = GraphNode::transform_to_impl(ancestors);

    CHECK_CALL(PREFIXED_API(GraphAddChildGraphNode)(
        &child.node,
        other.graph,
        ancestors.size() > 0 ? ancestors_impl.data() : nullptr, ancestors.size(),
        this->graph
    ));

    return child;
}

GraphExecutable::GraphExecutable(const Graph& graph)
{
    printf("> Instantiating an executable graph from %p.\n", graph.graph);
    CHECK_CALL(PREFIXED_API(GraphInstantiate)(&graph_exec, graph.graph, nullptr, nullptr, 0));
}

GraphExecutable::~GraphExecutable()
{
    printf("> Destroying executable graph at %p.\n", graph_exec);
    CHECK_CALL(PREFIXED_API(GraphExecDestroy)(graph_exec));
}

void GraphExecutable::set_enabled(const GraphNode& node, const bool is_enabled) const {
    CHECK_CALL(PREFIXED_API(GraphNodeSetEnabled)(graph_exec, node.node, is_enabled));
}

void GraphExecutable::submit(const Stream& stream)
{
    printf("> Submitting executable graph %p on stream %p.\n", graph_exec, stream.stream);
    CHECK_CALL(PREFIXED_API(GraphLaunch)(graph_exec, stream.stream));
}

std::vector<PREFIXED_API(GraphNode_t)> GraphNode::transform_to_impl(const std::vector<GraphNode>& ancestors)
{
    std::vector<PREFIXED_API(GraphNode_t)> ancestors_impl(ancestors.size());

    std::transform(
        ancestors.cbegin(), ancestors.cend(), ancestors_impl.begin(),
        [](const auto& anc) -> PREFIXED_API(GraphNode_t) { return anc.node; }
    );

    return ancestors_impl;
}

template <typename Functor>
GraphNodeKernel<Functor>::GraphNodeKernel(const Functor& functor, const size_t shape)
{
    printf("> Creating a kernel graph node with functor and size %zu (%s).\n", shape, __PRETTY_FUNCTION__);

    params = {};

    inputs.resize(1);
    inputs[0] = (void*)&functor;

    params.gridDim        = dim3(1,     1, 1);
    params.blockDim       = dim3(1, shape, 1);
    params.sharedMemBytes = 0;
    params.func           = (void * )get_driver<Functor>();
    params.kernelParams   = inputs.data();
    params.extra          = nullptr;
}

template <typename Functor>
void GraphNodeKernel<Functor>::add(const Graph& graph, const std::vector<GraphNode>& ancestors)
{
    printf("> Adding graph kernel node %p to graph %p with %zu ancestors.\n", node, graph.graph, ancestors.size());
    const auto ancestors_impl = transform_to_impl(ancestors);
    CHECK_CALL(PREFIXED_API(GraphAddKernelNode)(
        &node, graph.graph,
        (ancestors_impl.size() > 0 ? ancestors_impl.data() : nullptr), ancestors_impl.size(),
        &params
    ));
}

#if defined(GRAPH_DISPATCHING_ENABLE_CUDA) || defined(DOXYGEN)
GraphNodeConditionalIf::GraphNodeConditionalIf(cudaGraphConditionalHandle handle)
{
    printf("> Creating a graph conditional if node for handle %llu (%s).\n", handle, __PRETTY_FUNCTION__);

    params.type               = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type   = cudaGraphCondTypeIf;
    params.conditional.size   = 1;
}

void GraphNodeConditionalIf::add(const Graph& graph, const std::vector<GraphNode>& ancestors)
{
    printf("> Adding graph conditional node %p to graph %p with %zu ancestors.\n", node, graph.graph, ancestors.size());
    const auto ancestors_impl = transform_to_impl(ancestors);
    CHECK_CALL(PREFIXED_API(GraphAddNode)(
        &node, graph.graph,
        (ancestors_impl.size() > 0 ? ancestors_impl.data() : nullptr), ancestors_impl.size(),
        &params
    ));
}
#endif

template <typename Functor>
void GraphNodeHost<Functor>::driver(void* data)
{
    const auto& functor = static_cast<NodeHostCallbackData*>(data)->functor;
    functor.operator()();
}

template <typename Functor>
template <typename T> requires std::same_as<std::remove_cvref_t<T>, Functor>
GraphNodeHost<Functor>::GraphNodeHost(T&& functor)
{
    printf("> Creating a graph host node with functor (%s).\n", __PRETTY_FUNCTION__);

    data = {.functor = std::forward<T>(functor)};

    params          = {};
    params.fn       = driver;
    params.userData = &data;
}

template <typename Functor>
void GraphNodeHost<Functor>::add(const Graph& graph, const std::vector<GraphNode>& ancestors)
{
    printf("> Adding graph host node %p to graph %p with %zu ancestors.\n", node, graph.graph, ancestors.size());
    const auto ancestors_impl = transform_to_impl(ancestors);
    CHECK_CALL(PREFIXED_API(GraphAddHostNode)(
        &node, graph.graph,
        (ancestors_impl.size() > 0 ? ancestors_impl.data() : nullptr), ancestors_impl.size(),
        &params
    ));
}

template <typename T>
GraphNodeMemoryAllocation<T>::GraphNodeMemoryAllocation(const size_t size, const Stream& stream)
{
    const auto device_id = stream.device();

    printf("> Creating a graph memory node for device ID %d of size %zu of %s (%s).\n", device_id, size, typeid(T).name(), __PRETTY_FUNCTION__);

    /// See https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaMemAllocNodeParams.html#structcudaMemAllocNodeParams.
    /// For the pool properties, see https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaMemPoolProps.html#structcudaMemPoolProps.
    params.bytesize = size * sizeof(T);

    params.poolProps.allocType     = PREFIXED_API(MemAllocationTypePinned);
    params.poolProps.location.type = PREFIXED_API(MemLocationTypeDevice);
    params.poolProps.location.id   = device_id;
}

template <typename T>
void GraphNodeMemoryAllocation<T>::add(const Graph& graph, const std::vector<GraphNode>& ancestors)
{
    printf("> Adding graph memory allocation node %p to graph %p with %zu ancestors.\n", node, graph.graph, ancestors.size());
    const auto ancestors_impl = transform_to_impl(ancestors);
    CHECK_CALL(PREFIXED_API(GraphAddMemAllocNode)(
        &node, graph.graph,
        (ancestors_impl.size() > 0 ? ancestors_impl.data() : nullptr), ancestors_impl.size(),
        &params
    ));

    this->ptr = static_cast<T*>(params.dptr);
}

void GraphNodeMemoryFree::add(const Graph& graph, const std::vector<GraphNode>& ancestors)
{
    printf("> Adding graph memory free node %p to graph %p with %zu ancestors.\n", node, graph.graph, ancestors.size());
    const auto ancestors_impl = transform_to_impl(ancestors);
    CHECK_CALL(PREFIXED_API(GraphAddMemFreeNode)(
        &node, graph.graph,
        (ancestors_impl.size() > 0 ? ancestors_impl.data() : nullptr), ancestors_impl.size(),
        ptr
    ));
}

template <typename Functor> requires ( ! std::is_pointer_v<Functor> )
__global__ void driver(const Functor functor)
{
    int index = threadIdx.y;
    functor.operator()(index);
}

} // namespace tests::cuda

#endif // GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_DEF_HPP
