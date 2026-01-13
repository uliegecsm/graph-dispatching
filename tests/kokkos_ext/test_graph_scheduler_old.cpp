#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/kokkos_ext/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Treat @c Kokkos::Graph within the P2300 framework
 * -------------------------------------------------
 *
 * Check that we can mimic the scheduler-based programming from P2300 by
 * wrapping @c Kokkos::Graph. It's mainly done with
 * @ref Kokkos::Experimental::GraphContext.
 *
 * The tests can be found in @ref kokkos_ext/test_graph_scheduler_old.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

//! @test Check that @ref Kokkos::Experimental::GraphContext does its duty well.
TEST(GraphContext, then)
{
    using view_t = Kokkos::View<int[1], Kokkos::SharedSpace>;

    const execution_space exec {};

    const view_t data(Kokkos::view_alloc("data", exec));

    Kokkos::Experimental::GraphContext graph_ctx {exec};

    auto chain = Kokkos::Experimental::schedule(graph_ctx.get_scheduler())
        | Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy(0, 1),
            MyDummyFunctor{.data = data}
        )
        | Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy(0, 1),
            MyDummyFunctor{.data = data}
        );

    Kokkos::Experimental::graph::submit(exec, chain);

    exec.fence();

    ASSERT_EQ(data(0), 2);

    Kokkos::Experimental::graph::submit(exec, chain);

    exec.fence();

    ASSERT_EQ(data(0), 4);
}

/**
 * @test Check that @ref Kokkos::Experimental::GraphContext supports a many device
 *       case. The user can request a scheduler for a given device, much like the
 *       @c nvexec::stream_context (see https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream_context.cuh#L306).
 */
TEST(GraphContext, then_many_devices)
{
#if !defined(KOKKOS_ENABLE_CUDA)
    GTEST_SKIP() << "Only Cuda supports multi-GPU graph.";
#else
    int device_count = 0;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
    if(device_count < 2)
        GTEST_SKIP() << "You need at least 2 GPUs for this test.";
#endif

#if defined(KOKKOS_ENABLE_CUDA)
    using view_t = Kokkos::View<int[1], Kokkos::CudaHostPinnedSpace>;

    auto create_exec_on_device = [](const unsigned short int devID)
    {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devID));
        cudaStream_t stream = nullptr;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));
        return Kokkos::Cuda(stream, Kokkos::Impl::ManageStream::yes);
    };

    const Kokkos::Cuda exec_0 = create_exec_on_device(0), exec_1 = create_exec_on_device(1);

    const view_t data(Kokkos::view_alloc("data", exec_0));

    Kokkos::Experimental::GraphContext graph_ctx {exec_0, exec_1};

    auto chain = Kokkos::Experimental::schedule(graph_ctx.get_scheduler(0))
        | Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy(0, 1),
            MyDummyFunctor{.data = data}
        )
        //! @todo We're missing a continues on GPU 1 here.
        | Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy(0, 1),
            MyDummyFunctor{.data = data}
        );

    Kokkos::Experimental::graph::submit(exec_0, chain);

    exec_0.fence();

    ASSERT_EQ(data(0), 2);

    Kokkos::Experimental::graph::submit(exec_0, chain);

    exec_0.fence();

    ASSERT_EQ(data(0), 4);
#endif
}

} // namespace tests::kokkos_ext
