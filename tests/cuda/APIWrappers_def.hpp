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

void GraphExecutable::submit(const Stream& stream)
{
    printf("> Submitting executable graph %p on stream %p.\n", graph_exec, stream.stream);
    CHECK_CALL(PREFIXED_API(GraphLaunch)(graph_exec, stream.stream));
}

template <typename Functor> requires ( ! std::is_pointer_v<Functor> )
__global__ void driver(const Functor functor)
{
    int index = threadIdx.y;
    functor.operator()(index);
}

} // namespace tests::cuda

#endif // GRAPH_DISPATCHING_TESTS_CUDA_APIWRAPPERS_DEF_HPP
