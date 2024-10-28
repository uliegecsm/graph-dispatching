#include "gtest/gtest.h"

#include "tests/cuda/APIWrappers_def.hpp"
#include "tests/cuda/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Play with @c Cuda graph memory node
 * -----------------------------------
 *
 * This test shows how a @c Cuda graph memory node can be added.
 *
 * It must be noted that many restrictions apply to memory nodes. The most important
 * one is that the memory allocation size must be known when creating the graph, and cannot be
 * updated (see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#individual-node-update for instance).
 *
 * References:
 *  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-memory-nodes
 *
 * The test can be found in @ref cuda/test_memory_node.cpp.
 */

namespace tests::cuda
{

//! @test Check how a @c Cuda graph memory node can be added.
TEST(cuda, memory_node)
{
    using view_t    = View<int>;
    using functor_t = MyFunctor<view_t>;

    constexpr size_t size = 2<<8;

    Stream stream;

    Graph graph;

    GraphNodeMemoryAllocation<int> alloc(size, stream);
    alloc.add(graph);

    functor_t functor_d{.data = view_t(size, alloc.ptr)};
    GraphNodeKernel work(functor_d, size);
    work.add(graph, {alloc});

    GraphNodeMemoryFree free(alloc.ptr);
    free.add(graph, {work});

    GraphExecutable graph_exec(graph);

    graph_exec.submit(stream);
    stream.fence();
}

} // namespace tests::cuda
