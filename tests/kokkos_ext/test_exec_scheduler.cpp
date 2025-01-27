#include "gtest/gtest.h"

#include "kokkos_ext/Kokkos_Graph_Execution.hpp"

#include "tests/kokkos_ext/Helpers.hpp"

/**
 * @addtogroup unittests
 *
 * Treat @c Kokkos execution spaces within the P2300 framework
 * -----------------------------------------------------------
 *
 * Check that we can mimic the scheduler-based programming from P2300 by
 * wrapping @c Kokkos execution spaces. It's mainly done with
 * @ref Kokkos::Experimental::ExecutionSpaceContext.
 *
 * The tests can be found in @ref kokkos_ext/test_exec_scheduler.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

//! @test Check that @ref Kokkos::Experimental::ExecutionSpaceContext does its duty well.
TEST(ExecutionSpaceContext, then)
{
    using view_t = Kokkos::View<int[1], Kokkos::SharedSpace>;

    const execution_space exec {};

    view_t data(Kokkos::view_alloc("data", exec));

    Kokkos::Experimental::ExecutionSpaceContext esc{exec};

    auto sch = esc.get_scheduler();

    static_assert(std::same_as<decltype(sch), Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<Kokkos::Cuda>>);

    auto chain_0 = Kokkos::Experimental::schedule(sch);

    using sch_sender_t = Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler<Kokkos::Cuda>::Sender_;
    static_assert(std::same_as<decltype(chain_0), sch_sender_t>);

    auto partial = Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy(0, 1),
            MyDummyFunctor{.data = data}
        );

    using first_pfor_partial_t = Kokkos::Experimental::graph::details::PartialAlgorithm<
        Kokkos::ParallelForTag, std::string, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>
    >;
    static_assert(std::same_as<decltype(partial), first_pfor_partial_t>);

    auto chain_1 = std::move(chain_0) | std::move(partial);

    using first_pfor_sender_t = Kokkos::Experimental::graph::details::ParallelForSender<
        sch_sender_t, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>
    >;
    static_assert(std::same_as<decltype(chain_1), first_pfor_sender_t>);

    auto chain_2 = std::move(chain_1) 
        | Kokkos::Experimental::graph::parallel_for(
            Kokkos::RangePolicy(0, 1),
            MyDummyFunctor{.data = data}
        );

    using second_pfor_sender_t = Kokkos::Experimental::graph::details::ParallelForSender<
        first_pfor_sender_t, Kokkos::RangePolicy<>, MyDummyFunctor<view_t>
    >;
    static_assert(std::same_as<decltype(chain_2), second_pfor_sender_t>);

    Kokkos::Experimental::graph::sync_wait(std::move(chain_2));

    exec.fence();

    ASSERT_EQ(data(0), 2);
}

} // namespace tests::kokkos_ext
