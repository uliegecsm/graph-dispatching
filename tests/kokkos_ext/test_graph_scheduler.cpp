#include "gtest/gtest.h"

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

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
 * The tests can be found in @ref kokkos_ext/test_graph_scheduler.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

//! @test Check that @ref Kokkos::Experimental::GraphContext does its duty well.
TEST(GraphContext, then)
{
    using view_t = Kokkos::View<int[1], Kokkos::SharedSpace>;

    const execution_space exec {};

    view_t data(Kokkos::view_alloc("data", exec));

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
 * @test Check that @ref Kokkos::Experimental::GraphContext correctly requires that the user writes
 *       a @c continues_on after a @c when_all.
 */
TEST(GraphContext, when_all_continues_on)
{
    using view_t = Kokkos::View<int[1], Kokkos::SharedSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;

    const execution_space exec {};

    view_t data(Kokkos::view_alloc("data", exec));

    Kokkos::Experimental::GraphContext graph_ctx {exec};

    auto fork = Kokkos::Experimental::schedule(graph_ctx.get_scheduler()) | Kokkos::Experimental::graph::split();

    // TODO the follopwing code should either not compile or not put the last then node after the when all in the graph
    auto chain = Kokkos::Experimental::when_all(
        fork | Kokkos::Experimental::graph::parallel_for(Kokkos::RangePolicy(0, 1), MyDummyFunctor{.data = data}),
        fork | Kokkos::Experimental::graph::parallel_for(Kokkos::RangePolicy(0, 1), MyDummyFunctor{.data = data})
    ) | Kokkos::Experimental::graph::parallel_for(Kokkos::RangePolicy(0, 1), MyDummyFunctor{.data = data});

    Kokkos::Experimental::graph::submit(exec, chain);

    exec.fence();

    ASSERT_EQ(data(0), 3);

    Kokkos::Experimental::graph::submit(exec, chain);

    exec.fence();

    ASSERT_EQ(data(0), 6);
}

} // namespace tests::kokkos_ext
