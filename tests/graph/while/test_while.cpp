#include <concepts>
#include <span>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

/**
 * @addtogroup unittests
 *
 * Compare submitting a @c Kokkos::Graph with a manual while loop and a @c Cuda while node
 * ---------------------------------------------------------------------------------------
 *
 * The tests can be found in @ref tests/graph/while/test_while.cpp.
 */

namespace tests::graph::while_node
{

//! Work in a node.
template <typename ViewType>
struct Work
{
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index) const {
        data(index) += 1;
    }
};

//! Reduction node.
template <typename ViewType>
struct Reduce
{
    ViewType data;

    template <std::integral T>
    KOKKOS_FUNCTION
    void operator()(const T index, bool& current) const {
        current = current && data(index) >= 42;
    }
};

//! Convergence.
template <typename ViewType>
struct Convergence
{
    ViewType value;
    cudaGraphConditionalHandle handle;

    /// @note @c cudaGraphSetConditional is a pure @c __device__ function.
    ///       @c clang would complain that @ref Convergence is not invocable if marking the call operator as @c __device__, see
    ///       https://github.com/kokkos/kokkos/blob/7a516890899a0b0f80393915e27b5880eebf745e/core/src/Kokkos_GraphNode.hpp#L235-L236.
    ///       So it seems @c clang treats the attributes as part of the signature.
    ///       On the other hand, @c nvcc would complain if marking as @c KOKKOS_FUNCTION because it's a call to a @c __device__ function
    ///       from a @c __host__ @c __device__ function.
    ///       In such a narrowed case, the solution is to use @c KOKKOS_IF_ON_HOST and @c KOKKOS_IS_ON_DEVICE.
    KOKKOS_FUNCTION
    void operator()() const
    {
        KOKKOS_IF_ON_DEVICE(if(value()) cudaGraphSetConditional(handle, false);)
        KOKKOS_IF_ON_HOST(Kokkos::abort("device only");)
    }
};

class WhileTest : public ::testing::Test
{
public:
    using graph_t = Kokkos::Experimental::Graph<Kokkos::Cuda>;

    using data_t        = Kokkos::View<double*, Kokkos::CudaSpace>;
    using bool_t        = Kokkos::View<bool,    Kokkos::CudaSpace>;
    using shared_bool_t = Kokkos::View<bool,    Kokkos::SharedSpace>;

    static constexpr size_t size = 128<<1;

public:
    void SetUp() override
    {
        this->exec = Kokkos::Experimental::partition_space(Kokkos::Cuda{}, 1)[0];
        this->data = data_t(Kokkos::view_alloc(*exec, "data"), size);
    }

    void check() const
    {
        const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, this->data);
        ASSERT_THAT(
            (std::span{mirror.data(), mirror.size()}),
            ::testing::Each(::testing::Eq(42.))
        );
    }

protected:
    std::optional<Kokkos::Cuda> exec;
    data_t data;
};

//! @test The @c while loop is on host.
TEST_F(WhileTest, manual)
{
    const graph_t graph{Kokkos::Experimental::get_device_handle(*this->exec)};

    //! Add the parallel-for.
    const auto pfor = graph.root_node().then_parallel_for(
        Kokkos::Experimental::node_props(Kokkos::Experimental::get_device_handle(*this->exec)),
        Kokkos::RangePolicy<Kokkos::Cuda>(0, size),
        Work<data_t>{.data = this->data}
    );
    
    //! Add the reduction. The reduction target must be host and device accessible.
    const shared_bool_t value(Kokkos::view_alloc("shared bool"));

    pfor.then_parallel_reduce(
        Kokkos::Experimental::node_props(Kokkos::Experimental::get_device_handle(*this->exec)),
        Kokkos::RangePolicy<Kokkos::Cuda>(0, size),
        Reduce<data_t>{.data = this->data},
        Kokkos::LAnd<bool, Kokkos::CudaSpace>{value}
    );

    //! Loop and submit the graph until the condition is met. It requires fencing after submission before reading the conditional.
    while(!value())
    {
        graph.submit(*this->exec);
        this->exec->fence();
    }

    this->check();
}

//! @test The @c while loop is on device.
TEST_F(WhileTest, node)
{
    //! Create the outer graph, @c while node and handle in raw @c Cuda.
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphCreate(&graph, 0));

    cudaGraphConditionalHandle conditional_handle;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphConditionalHandleCreate(&conditional_handle, graph, 1, cudaGraphCondAssignDefault));

    cudaGraphNodeParams conditional_params = {};
    conditional_params.type                = cudaGraphNodeTypeConditional;
    conditional_params.conditional.handle  = conditional_handle;
    conditional_params.conditional.type    = cudaGraphCondTypeWhile;
    conditional_params.conditional.size    = 1;
    cudaGraphNode_t conditional_node = nullptr;

    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphAddNode(
        &conditional_node,
        graph, nullptr, 
#if CUDART_VERSION >= 13000
        nullptr,
#endif
        0, &conditional_params));
    {
        //! Create the graph from the @c while node subgraph.
        const graph_t inner{Kokkos::Experimental::get_device_handle(*this->exec), conditional_params.conditional.phGraph_out[0]};

        //! Add the parallel-for.
        const auto pfor = inner.root_node().then_parallel_for(
            Kokkos::Experimental::node_props(Kokkos::Experimental::get_device_handle(*this->exec)),
            Kokkos::RangePolicy<Kokkos::Cuda>(0, size),
            Work<data_t>{.data = this->data}
        );

        //! Add the reduction. The reduction result must be device accessible.
        bool_t value(Kokkos::view_alloc(*exec, "device bool"));

        const auto pred = pfor.then_parallel_reduce(
            Kokkos::Experimental::node_props(Kokkos::Experimental::get_device_handle(*this->exec)),
            Kokkos::RangePolicy<Kokkos::Cuda>(0, size),
            Reduce<data_t>{.data = this->data},
            Kokkos::LAnd<bool, Kokkos::CudaSpace>{value}
        );

        //! Add a node that will set the @c while conditional.
        pred.then(Kokkos::Experimental::node_props(
            "convergence",
            Kokkos::Experimental::get_device_handle(*this->exec)),
            Convergence<bool_t>{.value = std::move(value), .handle = conditional_handle});

        //! Create the executable graph and submit once. It will converge during the first submission since the @c while is embedded.
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphLaunch(graph_exec, this->exec->cuda_stream()));

        this->exec->fence();
    }

    //! Destroy graphs.
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphExecDestroy(graph_exec));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphDestroy    (graph));

    this->check();
}

} // namespace tests::graph::while_node
