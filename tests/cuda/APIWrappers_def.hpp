#ifndef GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_DEF_HPP
#define GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_DEF_HPP

#include "tests/cuda/APIWrappers.hpp"

namespace tests::cuda
{
Stream::Stream() {
    CHECK_CALL(PREFIXED_API(StreamCreateWithFlags)(&stream, PREFIXED_API(StreamNonBlocking)));
}

void Stream::fence() const {
    CHECK_CALL(PREFIXED_API(StreamSynchronize)(stream));
}

template <typename T>
View<T>::View(const Stream& stream, const size_t size_) : size(size_), owning(true)
{
    const auto raw_size = size * sizeof(T);
    printf("> Allocating data of size %zu, memset it to 0.\n", raw_size);
    CHECK_CALL(PREFIXED_API(MallocAsync)(&buffer,    raw_size, stream.stream));
    CHECK_CALL(PREFIXED_API(MemsetAsync)( buffer, 0, raw_size, stream.stream));
}

template <typename T>
auto View<T>::get_host_copy(const Stream& stream) const
{
    std::vector<T> host_copy(size);
    CHECK_CALL(PREFIXED_API(MemcpyAsync)(host_copy.data(), buffer, size * sizeof(T), PREFIXED_API(MemcpyDeviceToHost), stream.stream));
    return host_copy;
}

template <typename T>
View<T>::~View()
{
    if(owning)
    {
        printf("> Deallocating data of size %zu at %p.\n", size, buffer);
        CHECK_CALL(PREFIXED_API(Free)(buffer));
    }
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

void Graph::print(const char* path, const unsigned int flags)
{
    CHECK_CALL(cudaGraphDebugDotPrint(graph, path, flags));
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

auto GraphNode::transform_to_impl(const std::vector<GraphNode>& ancestors)
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

template <typename Functor> requires ( ! std::is_pointer_v<Functor> )
__global__ void driver(const Functor functor)
{
    int index = threadIdx.y;
    functor.operator()(index);
}

} // namespace tests::cuda

#endif // GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_DEF_HPP
